import unittest
from unittest.mock import Mock, patch

from EmotionDetection import emotion_detector


class TestEmotionDetector(unittest.TestCase):

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_joy(self, mock_post):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "emotionPredictions": [{
                "emotion": {
                    "anger": 0.01,
                    "disgust": 0.01,
                    "fear": 0.02,
                    "joy": 0.95,
                    "sadness": 0.01,
                }
            }]
        }
        mock_post.return_value = mock_response

        result = emotion_detector("I am happy")

        self.assertEqual(result["dominant_emotion"], "joy")
        self.assertEqual(result["joy"], 0.95)

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_sadness(self, mock_post):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "emotionPredictions": [{
                "emotion": {
                    "anger": 0.01,
                    "disgust": 0.02,
                    "fear": 0.03,
                    "joy": 0.04,
                    "sadness": 0.90,
                }
            }]
        }
        mock_post.return_value = mock_response

        result = emotion_detector("I am sad")

        self.assertEqual(result["dominant_emotion"], "sadness")

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_required_keys(self, mock_post):
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
        mock_post.return_value = mock_response

        result = emotion_detector("A sample sentence")

        expected = {
            "anger", "disgust", "fear",
            "joy", "sadness", "dominant_emotion"
        }
        self.assertEqual(set(result.keys()), expected)


if __name__ == "__main__":
    unittest.main()