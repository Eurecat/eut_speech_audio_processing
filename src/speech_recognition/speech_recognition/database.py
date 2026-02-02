import datetime
import logging
from typing import Optional

import numpy as np
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from scipy.spatial.distance import cosine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataBaseManager:
    """Manages MongoDB operations for speaker embeddings."""

    def __init__(
        self,
        mongo_uri: str = "mongodb://eurecat:cerdanyola@localhost:27017/?authSource=admin&serverSelectionTimeoutMS=5000",
    ):
        try:
            self.client = MongoClient(mongo_uri)
            self.client.admin.command("ping")
            self.db = self.client["speaker_recognition"]
            self.speakers = self.db["speakers"]
            self.speakers.create_index("speaker_name", unique=True)
            logger.info("Connected to MongoDB\n")
            logger.info(
                f"[INFO] Connected to MongoDB at {mongo_uri}, database: speaker_recognition, collection: speakers"
            )
        except ConnectionFailure:
            logger.error("Could not connect to MongoDB. Is it running?\n")
            raise

    def save_speaker(self, name: str, embedding: np.ndarray):
        """Saves a speaker in the database"""
        doc = {
            "speaker_name": name,
            "embedding": embedding.tolist(),
            "date": datetime.datetime.now(datetime.timezone.utc),
        }
        self.speakers.update_one({"speaker_name": name}, {"$set": doc}, upsert=True)

    def find_speaker(
        self, embedding: np.ndarray, loggefunc, threshold: float = 0.5
    ) -> Optional[str]:
        """Searches for a matching speaker embedding in the database"""
        all_speakers = list(self.speakers.find())

        if not all_speakers:
            return None

        best_name = None
        best_distance = float("inf")

        for speaker in all_speakers:
            saved_embedding = np.array(speaker["embedding"])
            distance = cosine(embedding, saved_embedding)
            loggefunc.info(f"\033[93mDistance to {speaker['speaker_name']}: {distance}\033[0m")

            if distance < best_distance:
                best_distance = distance
                best_name = speaker["speaker_name"]

        if best_distance < threshold:
            return best_name, best_distance

        return None

    def number_speakers(self):
        """Shows the number of speakers in the database"""
        speakers = list(self.speakers.find())
        logger.info(f"Speakers in database: {len(speakers)}")
        for s in speakers:
            logger.info(f"  - {s['speaker_name']} (saved: {s['date'].strftime('%Y-%m-%d %H:%M')})")
        return len(speakers)

    def close(self):
        self.client.close()
