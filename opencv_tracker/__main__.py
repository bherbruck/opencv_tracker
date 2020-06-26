from .models import Detector

if __name__ == '__main__':
    detector = Detector(video_source='media/faces.mkv',
                        cascade='haar_cascade/face.xml',
                        output='output/output.mp4',
                        scale_factor=1.1,
                        min_neighbors=5,
                        max_distance=25,
                        timeout=10)
    detector.run()
