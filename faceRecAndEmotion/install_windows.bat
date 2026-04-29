@echo off
echo 🐶 Installing Smart AI Dog Robot dependencies on Windows
echo =====================================================

echo 📦 Installing numpy...
pip install numpy==1.24.3

echo 📦 Installing OpenCV...
pip install opencv-python==4.8.1.78

echo 📦 Installing TensorFlow (this may take a while)...
pip install tensorflow==2.13.0

echo 📦 Installing remaining packages...
pip install keras==2.13.1
pip install mtcnn==0.1.1
pip install pandas==2.0.3
pip install Pillow==10.0.0
pip install gtts==2.3.2
pip install pygame==2.5.2
pip install scikit-learn==1.3.0
pip install scipy==1.11.1
pip install matplotlib==3.7.2
pip install deepface==0.0.79

echo ✅ All packages installed successfully!
pause