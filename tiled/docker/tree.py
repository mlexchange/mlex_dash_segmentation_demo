import sys

import dask.array
import dask.dataframe
import numpy as np
import pandas as pd
import uuid
import itertools

import json

import pymongo
from bson.objectid import ObjectId

import pyarrow as pa
import pyarrow.parquet as pq

import collections.abc

from dataclasses import dataclass

from tiled.utils import import_object, DictView
from tiled.trees.utils import IndexersMixin, UNCHANGED
from tiled.readers.dataframe import DataFrameAdapter

from tiled.query_registration import QueryTranslationRegistry, register

from tiled.trees.in_memory import Tree
from tiled.readers.dataframe import DataFrameAdapter

def deserialize_parquet(data):
  reader = pa.BufferReader(data)
  table = pq.read_table(reader)
  return table.to_pandas()

@register(name="raw_mongo")
@dataclass
class RawMongoQuery:
    """
    Run a MongoDB query against a given collection.
    """

    query: str  # We cannot put a dict in a URL, so this a JSON str.

    def __init__(self, query):
        if isinstance(query, collections.abc.Mapping):
            query = json.dumps(query)
        self.query = query

@register(name="element")
@dataclass
class ElementQuery:

    symbol: str
    edge: str

    def __init__(self, symbol, edge):
        self.symbol = symbol
        self.edge = edge

def _get_database(uri, username, password):
    if not pymongo.uri_parser.parse_uri(uri)["database"]:
        raise ValueError(
            f"Invalid URI: {uri!r} " f"Did you forget to include a database?"
        )
    else:
        client = pymongo.MongoClient(uri, username=username, password=password)
        return client.get_database()

class MongoCollectionTree(collections.abc.Mapping, IndexersMixin):

    # Define classmethods for managing what queries this Tree knows.
    query_registry = QueryTranslationRegistry()
    register_query = query_registry.register
    register_query_lazy = query_registry.register_lazy

    @classmethod
    def from_uri(
        cls,
        uri,
        username,
        password,
        collection_name,
        *,
        metadata=None,
        access_policy=None,
        authenticated_identity=None,
        ):

        db = _get_database(uri, username, password)
        collection = db.get_collection(collection_name)

        return cls(collection,
                   metadata=metadata,
                   access_policy=access_policy,
                   authenticated_identity=authenticated_identity)

    def __init__(self, 
            collection,
            metadata=None,
            access_policy=None,
            authenticated_identity=None,
            queries=None):

        self._collection = collection

        self._metadata = metadata or {}
        if isinstance(access_policy, str):
            access_policy = import_object(access_policy)
        if (access_policy is not None) and (
            not access_policy.check_compatibility(self)
        ):
            raise ValueError(
                f"Access policy {access_policy} is not compatible with this Tree."
            )
        self._access_policy = access_policy
        self._authenticated_identity = authenticated_identity

        self._queries = list(queries or [])

        super().__init__()

    @property
    def collection(self):
        return self._collection

    @property
    def access_policy(self):
        return self._access_policy

    @property
    def authenticated_identity(self):
        return self._authenticated_identity

    @property
    def metadata(self):
        "Metadata about this Tree."
        # Ensure this is immutable (at the top level) to help the user avoid
        # getting the wrong impression that editing this would update anything
        # persistent.
        return DictView(self._metadata)

    def _build_mongo_query(self, *queries):
        combined = self._queries + list(queries)
        if combined:
            return {"$and": combined}
        else:
            return {}

    def _build_dataset(self, doc):
        raise NotImplementedError

    def __len__(self):
        return self._collection.count_documents(self._build_mongo_query())

    def __getitem__(self, key):
        query = self._build_mongo_query({"metadata.common.uid" : key})
        doc = self._collection.find_one(query)
        if doc is None:
            raise KeyError(key)
        return self._build_dataset(doc)

    def __iter__(self):
        for doc in self._collection.find(self._build_mongo_query()):
            yield str(doc["metadata"]["common"]["uid"])

    def authenticated_as(self, identity):
        if self._authenticated_identity is not None:
            raise RuntimeError(
                f"Already authenticated as {self.authenticated_identity}"
            )
        if self._access_policy is not None:
            raise NotImplementedError

        tree = self.new_variation(authenticated_identity=identity)
        return tree

    def new_variation(
        self,
        authenticated_identity=UNCHANGED,
        queries=UNCHANGED
    ):
        if authenticated_identity is UNCHANGED:
            authenticated_identity = self._authenticated_identity
        if queries is UNCHANGED:
            queries = self._queries

        return type(self)(
            collection = self._collection,
            metadata = self._metadata,
            access_policy = self._access_policy,
            authenticated_identity = authenticated_identity,
            queries = queries
        )

    # The following three methods are used by IndexersMixin
    # to define keys_indexer, items_indexer, and values_indexer.

    def search(self, query):
        """
        Return a Tree with a subset of the mapping.
        """
        return self.query_registry(query, self)

    def _keys_slice(self, start, stop, direction):
        assert direction == 1, "direction=-1 should be handled by the client"
        skip = start or 0
        if stop is not None:
            limit = stop - skip
        else:
            limit = None

        for doc in self._collection.find(self._build_mongo_query()).skip(skip).limit(limit):
            _id = str(doc["metadata"]["common"]["uid"])
            yield _id


    def _items_slice(self, start, stop, direction):
        assert direction == 1, "direction=-1 should be handled by the client"
        skip = start or 0
        if stop is not None:
            limit = stop - skip
        else:
            limit = None

        for doc in self._collection.find(self._build_mongo_query()).skip(skip).limit(limit):
            _id = str(doc["metadata"]["common"]["uid"])
            dset = self._build_dataset(doc)
            yield (_id, dset)

    def _item_by_index(self, index, direction):
        assert direction == 1, "direction=-1 should be handled by the client"

        doc = next(self._collection.find(self._build_mongo_query()).skip(index).limit(1))
        _id = str(doc["metadata"]["common"]["uid"])
        dset = self._build_dataset(doc)
        return (_id, dset)

class MongoXASTree(MongoCollectionTree):
    def _build_dataset(self, doc):
        data = doc["data"]
        assert data["structure_family"] == "dataframe"
        assert data["media_type"] == "application/x-parquet"
        df = deserialize_parquet(data["blob"])
        metadata = doc["metadata"]
        return DataFrameAdapter.from_pandas(df, metadata=metadata, npartitions=1)

def run_raw_mongo_query(query, tree):
    return tree.new_variation(
        queries=tree._queries + [json.loads(query.query)],
    )

def run_element_query(query, tree):
    results = tree.query_registry(
            RawMongoQuery({"metadata.common.element.symbol" : query.symbol, "metadata.common.element.edge" : query.edge}),
            tree)
    return results

MongoXASTree.register_query(RawMongoQuery, run_raw_mongo_query)
MongoXASTree.register_query(ElementQuery, run_element_query)
