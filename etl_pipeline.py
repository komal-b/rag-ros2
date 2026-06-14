
from clearml import Task
from pymongo import MongoClient
from bs4 import BeautifulSoup
import requests
from youtube_transcript_api import YouTubeTranscriptApi
import logging
import pymongo

# MongoDB connection: scraped/extracted raw documents are stored here for later embedding
mongo_client = MongoClient('mongodb://llm_engineering:llm_engineering@127.0.0.1:27018/?authSource=admin')
db = mongo_client['rag_system']
collection = db['raw_data']

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def etl_pipeline():
    """
    Build the raw knowledge base for the RAG system:
      1. Scrape ROS2/Nav2/MoveIt2/Gazebo documentation, GitHub repos, and blog posts
      2. Pull transcripts from related YouTube tutorials
      3. Print a summary of everything stored in MongoDB
    Both `task_logger` (ClearML) and `print`/`logger` (stdout) are used so progress
    is visible both in the ClearML dashboard and in the local console.
    """
    # ClearML task tracks/logs this ETL run so progress and failures are visible in the ClearML dashboard
    task = Task.init(project_name="RAG_System", task_name="ETL Task")
    task_logger = task.get_logger()

    # Scrape a curated list of ROS2-related pages (official docs, GitHub repos, blog
    # tutorials) and store their visible text content in MongoDB as raw_data documents
    def extract_ros2_docs():
        ros2_docs_urls = [
        "https://www.ros.org/",
        "https://github.com/bmaxdk/ROS2-Nav2-with-SLAM-and-Navigation",
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
        "https://github.com/ros-planning/moveit_commander",
        "https://medium.com/robotics-zone/understanding-moveit2-and-ros-2-for-robotics-61d74832cd2a",
        "https://medium.com/ros2-basics/building-your-own-ros2-navigation-stack-from-scratch-cc9c6b6a32e6",
        "https://medium.com/robotics-zone/introducing-ros2-and-gazebo-simulation-for-robotics-3e4054d4f8f8",
        "https://medium.com/robotics-zone/using-ros2-and-gazebo-to-simulate-robots-in-a-vibrant-world-34a6a4f35b28",
        "https://docs.ros.org/en/humble/index.html",
        "https://navigation.ros.org/",
        "https://docs.nav2.org/configuration/index.html",
        "https://docs.nav2.org/tutorials/index.html",
        "https://github.com/ros-planning/moveit2_tutorials",



    ]
        collection.create_index("url", unique=True)  # Ensure URL uniqueness to avoid duplicates

        for url in ros2_docs_urls:
            try:
                response = requests.get(url)
                soup = BeautifulSoup(response.content, 'html.parser')
                # Strip HTML and keep only the visible text for embedding/fine-tuning
                text = soup.get_text()
                data = {'source': 'github', 'url': url, 'content': text}
                collection.update_one({'url': url}, {'$set': data}, upsert=True)  # Use upsert to avoid duplicates on re-runs

                task_logger.report_text(f"Extracted and stored data from {url}")
                print(f"Extracted and stored data from {url}")  # Use standard logger
            except Exception as e:
                task_logger.report_text(f"Failed to extract data from {url}: {e}")
                print(f"Failed to extract data from {url}: {e}")  # Use standard logger

    # Pull transcripts for a curated list of ROS2-related YouTube videos and store them in
    # MongoDB, giving the RAG system spoken/tutorial-style explanations in addition to docs
    def extract_youtube_videos():
        # Note: 'rtrGoGsMVlI&list' includes a leftover "&list" query param from a copied
        # playlist URL, so YouTubeTranscriptApi will likely fail to resolve that one ID
        youtube_video_ids = ['7TVWlADXwRw', 'rtrGoGsMVlI&list',"sVUKeHMBtpQ","Xbij9Tst-WA","jkoGkAd0GYk","QIyhbMksHGY","Xbij9Tst-WA","sVUKeHMBtpQ"]

        for video_id in youtube_video_ids:
            try:
                # Fetch the auto-generated/manual transcript and flatten it into one text blob
                transcript = YouTubeTranscriptApi.get_transcript(video_id)
                transcript_text = ' '.join([t['text'] for t in transcript])
                data = {'source': 'youtube', 'url': f"https://www.youtube.com/watch?v={video_id}", 'content': transcript_text}
                collection.update_one({'url': data['url']}, {'$set': data}, upsert=True)  # Use upsert to avoid duplicates on re-runs
                task_logger.report_text(f"Extracted and stored data from YouTube video {video_id}")
                print(f"Extracted and stored data from YouTube video {video_id}")
            except Exception as e:
                task_logger.report_text(f"Failed to extract data from YouTube video {video_id}: {e}")
                print(f"Failed to extract data from YouTube video {video_id}: {e}")


    # Sanity check: list every URL currently stored, to confirm what made it into MongoDB
    def print_ingested_urls():
        urls = [doc['url'] for doc in collection.find()]
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
