
from clearml import Task
from pymongo import MongoClient
from bs4 import BeautifulSoup
import requests
from youtube_transcript_api import YouTubeTranscriptApi
import logging

# MongoDB connection
mongo_client = MongoClient('mongodb://llm_engineering:llm_engineering@127.0.0.1:27017')
db = mongo_client['rag_system']
collection = db['raw_data']

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def etl_pipeline():
    # Initialize the task with the correct project name
    task = Task.init(project_name="RAG_System", task_name="ETL Task")
    task_logger = task.get_logger()

    def extract_ros2_docs():
        ros2_docs_urls = [
        "https://www.ros.org/",
        "https://github.com/ros-navigation/navigation2",
        "https://github.com/bmaxdk/ROS2-Nav2-with-SLAM-and-Navigation",
        "https://github.com/moveit/moveit2",
        "https://github.com/ros-simulation/gazebo_ros_pkgs",
        "https://docs.nav2.org/",
        "https://moveit.ai/",
        "https://gazebosim.org/home",
        "https://github.com/Medissaoui07/Autonomus-Navigation-Robot-ROS2",
        "https://github.com/ros-navigation/navigation2",
        "https://github.com/moveit/moveit2",
        "https://github.com/AndrejOrsula/ign_moveit2_examples/blob/master/docs/README.md",
        "https://github.com/oKermorgant/gz_moveit2_examples",
        "https://medium.com/schmiedeone/getting-started-with-ros2-part-1-d4c3b7335c71",
        "https://medium.com/@tetraengnrng/a-beginners-guide-to-ros2-29721dcf49c8",
        "https://medium.com/@CanyonLakeRobotics/tuning-the-ros2-nav2-stack-5b01f455e217",
        "https://medium.com/@thehummingbird/navigation-ros-1-vs-navigation-2-ros-2-12398b64cd",
        "https://medium.com/@kathybuilds/building-rescue-robots-3-trying-out-gazebo-e821c47b09d1",
        "https://medium.com/@devkapiltech/stepping-into-the-virtual-world-mastering-robotics-simulation-with-gazebo-acf6860f252e",
        "https://medium.com/@santoshbalaji/manipulation-with-moveit2-visualizing-robot-arm-in-simulation-1-8cd3a46d42b4",
        "https://kolkemboi.medium.com/simulate-6-dof-robot-arm-in-ros2-gazebo-and-moveit2-a171c7e9b0ad",
        "https://github.com/ros2/rosgraph",
        "https://github.com/ros/ros_comm",
        "https://github.com/ros-perception/vision_opencv",
        "https://github.com/ros-planning/moveit_commander"
        "https://medium.com/robotics-zone/understanding-moveit2-and-ros-2-for-robotics-61d74832cd2a",
        "https://medium.com/ros2-basics/building-your-own-ros2-navigation-stack-from-scratch-cc9c6b6a32e6",
        "https://medium.com/robotics-zone/introducing-ros2-and-gazebo-simulation-for-robotics-3e4054d4f8f8",
        "https://medium.com/robotics-zone/using-ros2-and-gazebo-to-simulate-robots-in-a-vibrant-world-34a6a4f35b28"







    ]

        for url in ros2_docs_urls:
            try:
                response = requests.get(url)
                soup = BeautifulSoup(response.content, 'html.parser')
                text = soup.get_text()
                data = {'source': 'github', 'url': url, 'content': text}
                collection.insert_one(data)
                task_logger.report_text(f"Extracted and stored data from {url}")
                print(f"Extracted and stored data from {url}")  # Use standard logger
            except Exception as e:
                task_logger.report_text(f"Failed to extract data from {url}: {e}")
                print(f"Failed to extract data from {url}: {e}")  # Use standard logger

    def extract_youtube_videos():
        youtube_video_ids = ['7TVWlADXwRw', 'rtrGoGsMVlI&list=PLgG0XDQqJckkSJDPhXsFU_RIqEh08nG0V',"sVUKeHMBtpQ","Xbij9Tst-WA","jkoGkAd0GYk","QIyhbMksHGY","Xbij9Tst-WA","sVUKeHMBtpQ"]

        for video_id in youtube_video_ids:
            try:
                transcript = YouTubeTranscriptApi.get_transcript(video_id)
                transcript_text = ' '.join([t['text'] for t in transcript])
                data = {'source': 'youtube', 'url': f"https://www.youtube.com/watch?v={video_id}", 'content': transcript_text}
                collection.insert_one(data)
                task_logger.report_text(f"Extracted and stored data from YouTube video {video_id}")
                print(f"Extracted and stored data from YouTube video {video_id}")
            except Exception as e:
                task_logger.report_text(f"Failed to extract data from YouTube video {video_id}: {e}")
                print(f"Failed to extract data from YouTube video {video_id}: {e}")


    def print_ingested_urls():
        urls = [doc['url'] for doc in collection.find()]
        print("Ingested URLs:")
        print("Ingested URLs:")
        for url in urls:
            logger.info(url)
            print(url)

    # Execute ETL steps
    extract_ros2_docs()
    extract_youtube_videos()
    print_ingested_urls()

    task.close()

# Directly execute the ETL pipeline locally
if __name__ == "__main__":
    etl_pipeline()
