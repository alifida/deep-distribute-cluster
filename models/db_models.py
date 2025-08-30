# db_models.py
from sqlalchemy import Column, Integer, String, ForeignKey, Text, TIMESTAMP
from sqlalchemy.dialects.postgresql import INET
from sqlalchemy.orm import relationship
from utils.db import Base


class TrainCluster(Base):
    __tablename__ = "train_cluster"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(300), nullable=False)
    status = Column(String(300), nullable=False)

    nodes = relationship("TrainClusterNode", back_populates="cluster")


class TrainClusterNode(Base):
    __tablename__ = "train_clusternode"

    id = Column(Integer, primary_key=True, index=True)
    node_type = Column(String(10), nullable=False)
    ip_address = Column(INET, nullable=False)
    port = Column(Integer, nullable=False)
    cluster_id = Column(Integer, ForeignKey("train_cluster.id"), nullable=True)

    cluster = relationship("TrainCluster", back_populates="nodes")


class TrainDatasetImg(Base):
    __tablename__ = "train_dataset_img"

    id = Column(Integer, primary_key=True, index=True)
    data_name = Column(String(300), nullable=False)
    data_path = Column(String(300), nullable=False)
    metainfo = Column(Text, nullable=False)
    processed_at = Column(TIMESTAMP, nullable=False)
    delete_at = Column(TIMESTAMP, nullable=True)
    status = Column(String(300), nullable=False)
    #user_id = Column(Integer, ForeignKey("auth_user.id"), nullable=False)
    extracted_path = Column(Text, nullable=False)
    data_path_test = Column(String(300), nullable=False)
    extracted_path_test = Column(Text, nullable=False)


class TrainTrainingJob(Base):
    __tablename__ = "train_training_job"

    id = Column(Integer, primary_key=True, index=True)
    job_name = Column(String(300), nullable=False)
    status = Column(String(300), nullable=False)
    started_at = Column(TIMESTAMP, nullable=True)
    ended_at = Column(TIMESTAMP, nullable=True)
    algo = Column(String(300), nullable=False)
    dataset_img_id = Column(Integer, ForeignKey("train_dataset_img.id"), nullable=False)
    #user_id = Column(Integer, ForeignKey("auth_user.id"), nullable=False)
    result = Column(Text, nullable=True)
    parameter_settings = Column(Text, nullable=True)
    training_log = Column(Text, nullable=True)
    training_log_history = Column(Text, nullable=True)
