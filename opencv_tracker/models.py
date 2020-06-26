import cv2

from simple_tracker import Tracker


class Detection:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.cx = x + (w // 2)
        self.cy = y + (h // 2)


class Detector:
    def __init__(self, video_source=0, cascade='haar_cascade/frontalface.xml',
                 scale_factor=1.3, min_neighbors=5, max_distance=150, timeout=40,
                 output='output.avi'):
        self.cascade = cascade
        self.video_source = video_source
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.max_distance = max_distance
        self.timeout = timeout
        self.output = output

    def scale_imgae(img, scale=0.60):
        scale_percent = 60 # percent of original size
        width = int(img.shape[1] * scale_percent)
        height = int(img.shape[0] * scale_percent)
        dim = (width, height)
        return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    def run(self):
        cascade = cv2.CascadeClassifier(self.cascade)
        tracker = Tracker(max_distance=self.max_distance, timeout=self.timeout)
        cap = cv2.VideoCapture(self.video_source)
        out = cv2.VideoWriter(self.output, cv2.VideoWriter_fourcc(*"MP4V"), 10.0,
                              (int(cap.get(3)), int(cap.get(4))))

        while cap.isOpened():
            ret, img = cap.read()
            
            if ret:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                objects = [Detection(x, y, w, h)
                        for x, y, w, h in cascade.detectMultiScale(img,
                                                                    self.scale_factor,
                                                                    self.min_neighbors)]

                counts = tracker.update([(obj.cx, obj.cy) for obj in objects])
                for obj in objects:
                    cv2.rectangle(img, (obj.x, obj.y), (obj.x + obj.w,
                                                        obj.y + obj.h), (255, 0, 0), 2)

                for id, point in tracker.points.items():
                    text = str(id)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    scale = 0.5
                    textsize = cv2.getTextSize(text, font, scale, 2)[0]
                    textX = point[0] - (textsize[0] // 2)
                    textY = point[1] + (textsize[1] // 2)
                    cv2.putText(img, text, (textX, textY), font, scale, (255, 0, 0), 2)
                    
                out.write(img)
                cv2.imshow('video', img)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
                
        out.release()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    detector = Detector(video_source='media/faces.mkv',
                        cascade='haar_cascade/face.xml',
                        output='output/output.mp4',
                        scale_factor=1.1,
                        min_neighbors=5,
                        max_distance=25,
                        timeout=10)
    detector.run()