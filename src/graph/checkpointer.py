# src/graph/checkpointer.py
import logging
from typing import Any, Optional, AsyncIterator, List, Tuple, Dict
import datetime
from motor.motor_asyncio import AsyncIOMotorClient
from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint, CheckpointMetadata, CheckpointTuple
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langchain_core.runnables import RunnableConfig
from config.settings import settings

logger = logging.getLogger(__name__)

class MongoDBSaver(BaseCheckpointSaver):
    """MongoDB-based checkpoint saver for LangGraph workflows."""
    
    serde = JsonPlusSerializer()

    def __init__(
        self,
        mongo_uri: str = settings.MONGODB_URI,
        db_name: str = settings.MONGO_DATABASE_NAME,
        collection_name: str = settings.LANGGRAPH_CHECKPOINTS_COLLECTION,
    ):
        super().__init__(serde=self.serde)
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.collection_name = collection_name
        
        self.client = AsyncIOMotorClient(mongo_uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        logger.info(f"MongoDBSaver initialized: {db_name}.{collection_name}")

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Retrieve a checkpoint tuple from MongoDB."""
        configurable = config.get("configurable", {})
        thread_id = configurable.get("thread_id")
        if not thread_id:
            raise ValueError("thread_id must be present in config['configurable']")
        
        query: Dict[str, Any] = {"thread_id": thread_id}
        if thread_ts := configurable.get("thread_ts"):
            query["thread_ts"] = thread_ts
            sort_order = None
        else:
            sort_order = [("thread_ts", -1)]  # Get latest if no thread_ts specified
        
        doc = await self.collection.find_one(query, sort=sort_order)
        if not doc:
            return None

        checkpoint = self.serde.loads(doc["checkpoint"])
        metadata = self.serde.loads(doc["metadata"])
        
        parent_config = None
        if parent_ts := doc.get("parent_ts"):
            parent_config = {
                "configurable": {
                    "thread_id": thread_id,
                    "thread_ts": parent_ts,
                }
            }
        
        return CheckpointTuple(config, checkpoint, metadata, parent_config)

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """List checkpoints matching the given criteria."""
        query: Dict[str, Any] = {}
        
        if config and (thread_id := config.get("configurable", {}).get("thread_id")):
            query["thread_id"] = thread_id
        
        if filter:
            query.update({f"metadata.{k}": v for k, v in filter.items()})

        if before and (thread_ts := before.get("configurable", {}).get("thread_ts")):
            query["thread_ts"] = {"$lt": thread_ts}

        cursor = self.collection.find(query).sort("thread_ts", -1)
        if limit:
            cursor = cursor.limit(limit)

        async for doc in cursor:
            doc_config: RunnableConfig = {
                "configurable": {
                    "thread_id": doc["thread_id"],
                    "thread_ts": doc["thread_ts"],
                }
            }
            checkpoint = self.serde.loads(doc["checkpoint"])
            metadata = self.serde.loads(doc["metadata"])
            
            parent_config = None
            if parent_ts := doc.get("parent_ts"):
                parent_config = {
                    "configurable": {
                        "thread_id": doc["thread_id"],
                        "thread_ts": parent_ts,
                    }
                }
            
            yield CheckpointTuple(doc_config, checkpoint, metadata, parent_config)

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        parent_config: Optional[RunnableConfig] = None
    ) -> RunnableConfig:
        """Save a checkpoint to MongoDB."""
        configurable = config.get("configurable", {})
        thread_id = configurable.get("thread_id")
        if not thread_id:
            raise ValueError("thread_id must be present in config['configurable']")
        
        thread_ts = checkpoint.get("id")
        if not thread_ts:
            raise ValueError("checkpoint must have an 'id' (thread_ts)")

        doc_to_save: Dict[str, Any] = {
            "thread_id": thread_id,
            "thread_ts": thread_ts,
            "checkpoint": self.serde.dumps(checkpoint),
            "metadata": self.serde.dumps(metadata),
        }

        # Extract parent_ts from parent_config if provided
        parent_ts = None
        if parent_config:
            parent_configurable = parent_config.get("configurable", {})
            parent_ts = parent_configurable.get("thread_ts") or parent_config.get("id")
        
        # Fallback: use current config's thread_ts if different from checkpoint id
        if not parent_ts:
            current_thread_ts = configurable.get("thread_ts")
            if current_thread_ts and current_thread_ts != thread_ts:
                parent_ts = current_thread_ts
        
        if parent_ts:
            doc_to_save["parent_ts"] = parent_ts

        await self.collection.update_one(
            {"thread_id": thread_id, "thread_ts": thread_ts},
            {"$set": doc_to_save},
            upsert=True,
        )
        
        return {
            "configurable": {
                "thread_id": thread_id,
                "thread_ts": thread_ts,
            }
        }

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: List[Tuple[str, Any]],
        task_id: str,
    ) -> None:
        """Apply writes to an existing checkpoint."""
        configurable = config.get("configurable", {})
        if not isinstance(configurable, dict):
            raise ValueError("Config must contain a 'configurable' dictionary")
        
        thread_id = configurable.get("thread_id")
        if not thread_id:
            raise ValueError("thread_id must be present in config['configurable']")
        
        query: Dict[str, Any] = {"thread_id": thread_id}
        sort_order = None
        
        if thread_ts := configurable.get("thread_ts"):
            query["thread_ts"] = thread_ts
        else:
            sort_order = [("thread_ts", -1)]  # Get latest if no thread_ts specified

        doc = await self.collection.find_one(query, sort=sort_order)
        if not doc:
            logger.warning(f"No checkpoint found for writes (task_id: {task_id})")
            return

        try:
            checkpoint: Checkpoint = self.serde.loads(doc["checkpoint"])
            
            # Ensure channel structures exist
            checkpoint.setdefault("channel_values", {})
            checkpoint.setdefault("channel_versions", {})
            
            # Apply writes
            for channel, value in writes:
                checkpoint["channel_values"][channel] = value
                checkpoint["channel_versions"][channel] = checkpoint["channel_versions"].get(channel, 0) + 1
            
            checkpoint["ts"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            
            await self.collection.update_one(
                {"_id": doc["_id"]},
                {"$set": {"checkpoint": self.serde.dumps(checkpoint)}}
            )

        except Exception as e:
            logger.error(f"Error persisting writes (task_id: {task_id}): {e}", exc_info=True)

    async def aclose(self):
        """Close the MongoDB client connection."""
        if self.client:
            self.client.close()

    def __getstate__(self):
        """Prepare instance for pickling (exclude MongoDB connections)."""
        state = self.__dict__.copy()
        for key in ['client', 'db', 'collection']:
            state.pop(key, None)
        return state

    def __setstate__(self, state):
        """Reinitialize MongoDB connections after unpickling."""
        self.__dict__.update(state)
        self.client = AsyncIOMotorClient(self.mongo_uri)
        self.db = self.client[self.db_name]
        self.collection = self.db[self.collection_name]


if __name__ == "__main__":
    import asyncio
    import time
    from langchain_core.messages import HumanMessage, AIMessage 

    async def test_checkpointer():
        from config.logging_config import setup_logging 
        setup_logging(level="DEBUG")
        logger.info("--- Testing MongoDBSaver ---")

        if not settings.MONGODB_URI:
            logger.warning("MONGODB_URI not configured. Ensure MongoDB is running for testing.")

        test_collection_name = f"test_checkpoints_{int(time.time())}" 
        logger.info(f"Using test collection: {test_collection_name}")
        
        saver = MongoDBSaver(collection_name=test_collection_name)

        try:
            thread_id_1 = "thread_test_1"
            config1_v1: RunnableConfig = {"configurable": {"thread_id": thread_id_1}}
            
            ts_v1 = datetime.datetime.now(datetime.timezone.utc).isoformat() + "_v1"
            checkpoint1_v1: Checkpoint = {
                "v": 1, "id": ts_v1, "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "channel_values": {"messages": [HumanMessage(content="Hello from v1")]},
                "channel_versions": {"messages": 1}, "versions_seen": {"messages": {}},
            }
            metadata1_v1: CheckpointMetadata = {"source": "test", "step": 1}
            
            saved_config1_v1 = await saver.aput(config1_v1, checkpoint1_v1, metadata1_v1, parent_config=None)
            logger.info(f"Saved v1 checkpoint. Returned config: {saved_config1_v1}")
            assert saved_config1_v1["configurable"]["thread_ts"] == ts_v1

            retrieved_tuple1_v1 = await saver.aget_tuple(saved_config1_v1)
            assert retrieved_tuple1_v1 is not None
            assert retrieved_tuple1_v1.parent_config is None 

            parent_config_for_v2 = saved_config1_v1 
            config1_v2_for_put: RunnableConfig = {"configurable": {"thread_id": thread_id_1}}

            ts_v2 = datetime.datetime.now(datetime.timezone.utc).isoformat() + "_v2"
            checkpoint1_v2: Checkpoint = {
                "v": 1, "id": ts_v2, "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "channel_values": {"messages": [HumanMessage(content="Hello from v1"), AIMessage(content="Hi from v2")]},
                "channel_versions": {"messages": 2}, "versions_seen": {"messages": {}},
            }
            metadata1_v2: CheckpointMetadata = {"source": "test", "step": 2}
            
            saved_config1_v2 = await saver.aput(config1_v2_for_put, checkpoint1_v2, metadata1_v2, parent_config=parent_config_for_v2) 
            logger.info(f"Saved v2 checkpoint. Returned config: {saved_config1_v2}")

            retrieved_tuple1_v2 = await saver.aget_tuple(saved_config1_v2)
            assert retrieved_tuple1_v2 is not None
            assert retrieved_tuple1_v2.parent_config is not None
            assert retrieved_tuple1_v2.parent_config.get("configurable", {}).get("thread_ts") == ts_v1
            logger.info(f"Parent config of v2 points to: {retrieved_tuple1_v2.parent_config.get('configurable',{}).get('thread_ts')}") # type: ignore
            
            # Test with a malformed parent_config (missing 'configurable')
            malformed_parent_config: Any = {"thread_id": "malformed_parent", "thread_ts": "ts_malformed"}
            ts_v3 = datetime.datetime.now(datetime.timezone.utc).isoformat() + "_v3"
            checkpoint1_v3: Checkpoint = {"v": 1, "id": ts_v3, "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(), "channel_values": {"messages": [AIMessage(content="V3")]}} # type: ignore
            metadata1_v3: CheckpointMetadata = {"source": "test", "step": 3}
            config_v3_for_put: RunnableConfig = {"configurable": {"thread_id": thread_id_1}}
            
            logger.info(f"Attempting to save v3 with malformed parent_config: {malformed_parent_config}")
            saved_config1_v3 = await saver.aput(config_v3_for_put, checkpoint1_v3, metadata1_v3, parent_config=malformed_parent_config)
            retrieved_tuple1_v3 = await saver.aget_tuple(saved_config1_v3)
            assert retrieved_tuple1_v3 is not None
            # Depending on fallback logic, parent_ts might be None or from config_v3_for_put if it had a thread_ts.
            # In this specific test, config_v3_for_put has no thread_ts, so parent_ts should be None for v3.
            doc_v3 = await saver.collection.find_one({"thread_id": thread_id_1, "thread_ts": ts_v3})
            assert "parent_ts" not in doc_v3 if doc_v3 else False # Ensure parent_ts was not set from malformed
            logger.info(f"Saved v3 checkpoint. Document in DB (for parent_ts check): {doc_v3}")


            logger.info(f"\n--- Listing checkpoints for thread_id: {thread_id_1} ---")
            count = 0
            async for item in saver.alist(config={"configurable": {"thread_id": thread_id_1}}):
                count +=1
            assert count == 3, f"Expected 3 checkpoints, got {count}"

            logger.info("MongoDBSaver tests passed with robust parent_config access.")

        except Exception as e:
            logger.error(f"Error during MongoDBSaver test: {e}", exc_info=True)
        finally:
            if saver and hasattr(saver, 'client') and saver.client and hasattr(saver, 'db') and saver.db: 
                logger.info(f"Dropping test collection: {test_collection_name}")
                if saver.db : 
                    await saver.db.drop_collection(test_collection_name)
            if saver:
                await saver.aclose() 

    asyncio.run(test_checkpointer())