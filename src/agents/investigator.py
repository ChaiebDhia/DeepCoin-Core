"""
The Visual Investigator Agent - Attribute Expert
Handles unknown or highly degraded coins using Vision-Language Models.
"""

from typing import Dict


class VisualInvestigator:
    """
    Uses VLM (GPT-4o-mini) to describe coins when CNN confidence is low.
    Ensures users never receive empty "I don't know" responses.
    """
    
    def __init__(self, openai_api_key: str):
        self.api_key = openai_api_key
        # TODO: Initialize OpenAI client
    
    def investigate(self, image_path: str, cnn_prediction: Dict) -> Dict:
        """
        Analyze coin visual attributes when classification is uncertain.
        
        Args:
            image_path: Path to coin image
            cnn_prediction: CNN output with low confidence (<0.40)
        
        Returns:
            {
                "visual_description": str,
                "detected_features": {
                    "metal_color": str,
                    "profile_direction": str,
                    "inscriptions": list,
                    "symbols": list
                },
                "confidence_explanation": str
            }
        """
        # TODO: Implement in Phase 5
        # 1. Load image
        # 2. Call GPT-4o-mini Vision API
        # 3. Parse response for structured attributes
        
        return {
            "visual_description": "Placeholder: VLM will describe coin here",
            "detected_features": {},
            "confidence_explanation": "CNN uncertainty due to high wear/corrosion"
        }
    
    def generate_prompt(self, cnn_prediction: Dict) -> str:
        """Generate VLM prompt for coin description."""
        return f"""
        You are a numismatic expert analyzing an ancient coin.
        
        The AI classifier is uncertain (confidence: {cnn_prediction['confidence']:.2%}).
        Please describe what you observe:
        
        1. Metal appearance (color, patina)
        2. Profile direction and features
        3. Any visible inscriptions (even partial)
        4. Symbols or imagery
        5. Condition assessment
        
        Be precise and objective.
        """


if __name__ == "__main__":
    print("🔍 Visual Investigator Agent - Ready for Phase 5")
