import unittest
from unittest.mock import Mock, patch

from EmotionDetection.emotion_detection import emotion_detector


class TestEmotionDetector(unittest.TestCase):

    def check_dominant_emotion(self, dominant):
        scores = {
            "anger": 0.05,
            "disgust": 0.05,
            "fear": 0.05,
            "joy": 0.05,
            "sadness": 0.05,
        }
        scores[dominant] = 0.80

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "emotionPredictions": [
                {"emotion": scores}
            ]
        }

        with patch(
            "EmotionDetection.emotion_detection.requests.post",
            return_value=mock_response
        ):
            result = emotion_detector("Sample text for testing")

        self.assertEqual(result["dominant_emotion"], dominant)
        self.assertEqual(result[dominant], 0.80)

    def test_anger(self):
        self.check_dominant_emotion("anger")

    def test_disgust(self):
        self.check_dominant_emotion("disgust")

    def test_fear(self):
        self.check_dominant_emotion("fear")

    def test_joy(self):
        self.check_dominant_emotion("joy")

    def test_sadness(self):
        self.check_dominant_emotion("sadness")

    def test_required_keys(self):
        expected = {
            "anger", "disgust", "fear",
            "joy", "sadness", "dominant_emotion"
        }

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "emotionPredictions": [{
                "emotion": {
                    "anger": 0.1,
                    "disgust": 0.1,
                    "fear": 0.1,
                    "joy": 0.6,
                    "sadness": 0.1,
                }
            }]
        }

        with patch(
            "EmotionDetection.emotion_detection.requests.post",
            return_value=mock_response
        ):
            result = emotion_detector("Sample text")

        self.assertEqual(set(result.keys()), expected)


if __name__ == "__main__":
    unittest.main()

