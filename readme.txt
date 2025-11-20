GameCam AI

GameCam AI is an open-source web application designed for hunters and wildlife enthusiasts. It allows users to manage trail cameras, upload images, and uses AI to automatically detect and tag animals (Deer, Hogs, Coyote, etc.).

Features

AI Detection: Automatically identifies wildlife using computer vision.

Prime Time Predictor: Analyzes historical data to predict animal movement times.

Camera Management: Organize images by specific camera locations.

Data Visualization: Charts and graphs for animal activity patterns.

Free & Open Source: No subscription fees.

Tech Stack

Backend: Python, Flask, Celery, Redis

Database: MySQL / MariaDB

AI/ML: YOLOv5

Frontend: HTML, Tailwind CSS, JavaScript

Installation

Clone the repository:

git clone [https://github.com/YOUR_USERNAME/gamecam-ai.git](https://github.com/YOUR_USERNAME/gamecam-ai.git)


Install dependencies:

pip install -r requirements.txt


Set up your .env file with your database credentials and secret keys.

Run the application:

flask run


License & Credits

License

This project is licensed under the GNU General Public License v3.0 (GPL-3.0). See the LICENSE file for details.

Credits

YOLOv5: Object detection models provided by Ultralytics.

GameCam AI: Developed by JBroTX AI Solutions LLC.

Support the Project

If you find this software useful, consider supporting its development:
Donate Here