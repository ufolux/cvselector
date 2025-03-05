import os
import json
import argparse
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, Any
from dashscope import MultiModalConversation


class QwenUIFinder:
    def __init__(self, api_key, model_name="qwen-vl-plus"):
        """Initialize the QwenUIFinder with the DashScope API key."""
        self.api_key = api_key
        self.model_name = model_name
        print(f"Using model: {model_name} via DashScope API (MultiModalConversation)")
        
        # Set environment variable for API key
        os.environ['DASHSCOPE_API_KEY'] = api_key

    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze an image to identify UI elements using DashScope API.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Dictionary containing the identified UI elements
        """
        # Format the image path for DashScope
        local_path = os.path.abspath(image_path)
        image_path_url = f"file://{local_path}"
        
        # Prepare the prompt for Qwen with specific JSON format instructions
        prompt = """Find all buttons, inputs, and interactive elements on this screen. 
        
        Return the result in the following JSON format:
                
        {
            "elements": [
                {
                    "type": "button|input|text",
                    "coordinates": {
                        "x1": 0,  // top-left x coordinate (integer)
                        "y1": 0,  // top-left y coordinate (integer)
                        "x2": 0,  // bottom-right x coordinate (integer)
                        "y2": 0   // bottom-right y coordinate (integer)
                    },
                    "label_text": "text shown on the element",
                    "font_family": "font name if identifiable",
                    "font_size": "estimated font size in pixels"
                }
            ]
        }
        
        Coordinates must be numbers representing the bounding box in the format expected by PIL.ImageDraw.rectangle():
        - x1, y1: top-left corner coordinates
        - x2, y2: bottom-right corner coordinates
        - All coordinates must be positive integers
        - Ensure x2 > x1 and y2 > y1
        
        Only return valid JSON string that can be parsed directly. Do not include any explanations or markdown formatting in your response."""

        # Make the API call using MultiModalConversation
        print("Sending request to DashScope API...")
        try:
            messages = [
                {
                    "role": "system",
                    "content": [{"text": "You are an excellent UI selector. You will only return valid JSON string that can be parsed directly."}]
                },
                {
                    "role": "user",
                    "content": [
                        {"image": image_path_url},
                        {"text": prompt}
                    ]
                }
            ]
            
            response = MultiModalConversation.call(
                api_key=self.api_key,
                model=self.model_name,
                top_p=0.4,
                messages=messages
            )
            
            # Extract the response content
            if response and "output" in response and "choices" in response["output"] and len(response["output"]["choices"]) > 0:
                response_content = response["output"]["choices"][0]["message"]["content"][0]["text"]
                print("Text response:", response_content)
                
                # Extract JSON data from the text response using the helper function
                ui_elements = self._extract_json_from_text(response_content)
                return ui_elements
            else:
                print("Unexpected response format:", response)
                return {"error": "Unexpected response format from API"}
                
        except Exception as e:
            print(f"API request failed: {str(e)}")
            return {"error": f"API request failed: {str(e)}"}

    def _extract_json_from_text(self, text: str) -> Dict[str, Any]:
        """
        Extract JSON data from a text response, handling potential formatting issues.
        
        Args:
            text: Text response from the API that may contain JSON
            
        Returns:
            Extracted JSON data as a dictionary
        """
        try:
            # Clean up the response to remove potential comments
            import re
            # Remove JavaScript-style comments (both single-line and multi-line)
            clean_response = re.sub(r'//.*?(\n|$)|/\*.*?\*/', '', text, flags=re.DOTALL)
            
            # Find JSON structure in the cleaned response
            json_start = clean_response.find('{')
            json_end = clean_response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                # Extract JSON substring
                json_str = clean_response[json_start:json_end]
                return json.loads(json_str)
            else:
                # Fallback: try to parse the entire cleaned response
                return json.loads(clean_response)
                
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON from response: {e}")
            print("Response text:", text)
            return {"error": "Failed to parse JSON from model response"}

    def draw_elements(self, image_path: str, ui_elements: Dict[str, Any], output_path: str = None) -> str:
        """
        Draw rectangles around identified UI elements.

        Args:
            image_path: Path to the input image
            ui_elements: Dictionary containing UI element data
            output_path: Path to save the output image (optional)

        Returns:
            Path to the saved output image
        """
        if "error" in ui_elements:
            print(f"Error: {ui_elements['error']}")
            return None

        # Load the image
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)

        # Set up colors for different element types
        colors = {
            "button": (255, 0, 0, 128),  # Red
            "input": (0, 255, 0, 128),   # Green
            "text": (0, 0, 255, 128),    # Blue
            "default": (255, 255, 0, 128)  # Yellow
        }

        # Try different potential formats the model might return
        elements = []
        if "elements" in ui_elements:
            elements = ui_elements["elements"]
        elif isinstance(ui_elements, list):
            elements = ui_elements
        else:
            # If structure is unknown, try to infer it
            for key, value in ui_elements.items():
                if isinstance(value, list):
                    elements = value
                    break

            # If still empty, use the entire dict as a single element
            if not elements:
                elements = [ui_elements]

        # Draw rectangle for each element
        font = ImageFont.load_default()
        for idx, element in enumerate(elements):
            try:
                # Handle different coordinate formats
                if "coordinates" in element:
                    coords = element["coordinates"]
                    if isinstance(coords, list) and len(coords) == 4:
                        x1, y1, x2, y2 = coords
                    elif isinstance(coords, dict):
                        x1 = coords.get("x1", coords.get("left", 0))
                        y1 = coords.get("y1", coords.get("top", 0))
                        x2 = coords.get("x2", coords.get("right", 0))
                        y2 = coords.get("y2", coords.get("bottom", 0))
                else:
                    x1 = element.get("x1", element.get("left", 0))
                    y1 = element.get("y1", element.get("top", 0))
                    x2 = element.get("x2", element.get("right", 0))
                    y2 = element.get("y2", element.get("bottom", 0))

                element_type = element.get(
                    "type", element.get("widget_type", "default"))
                color = colors.get(element_type.lower(), colors["default"])

                # Draw rectangle
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

                # Draw label
                label_text = element.get("text", element.get(
                    "label_text", f"Element {idx+1}"))
                draw.text(
                    (x1, y1-15), f"{element_type}: {label_text}", fill=color[:3], font=font)

                # Print element details
                print(
                    f"Element {idx+1}: {element_type} at {(x1, y1, x2, y2)} with text '{label_text}'")

            except Exception as e:
                print(f"Error processing element {idx}: {e}")
                print(f"Element data: {element}")

        # Save the output image
        if not output_path:
            filename, ext = os.path.splitext(image_path)
            output_path = f"{filename}_annotated{ext}"
        else:
            # Check if output_path is a directory
            if os.path.isdir(output_path) or not os.path.exists(output_path) and not os.path.splitext(output_path)[1]:
                # Ensure directory exists
                os.makedirs(output_path, exist_ok=True)
                # Create filename based on input image name
                base_filename = os.path.basename(image_path)
                filename, ext = os.path.splitext(base_filename)
                output_path = os.path.join(
                    output_path, f"{filename}_annotated{ext}")

        image.save(output_path)
        print(f"Annotated image saved to {output_path}")
        return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Analyze UI elements in an image using Qwen-VL via DashScope API")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    parser.add_argument("--output", "-o", type=str,
                        default=None, help="Path to save the output image")
    parser.add_argument("--model", "-m", type=str,
                        default="qwen-vl-max", help="Qwen model to use")
    parser.add_argument("--api-key", "-k", type=str, help="DashScope API key")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug output")
    args = parser.parse_args()

    # Check if image exists
    if not os.path.exists(args.image_path):
        print(f"Error: Image not found at {args.image_path}")
        return

    # Get API key from arguments or environment variable
    api_key = args.api_key or os.environ.get("DASHSCOPE_API_KEY")

    if not api_key:
        print("Error: API key is required. Provide it with --api-key or set DASHSCOPE_API_KEY environment variable.")
        return

    # Initialize the UI finder
    ui_finder = QwenUIFinder(api_key=api_key, model_name=args.model)

    try:
        # Analyze the image
        ui_elements = ui_finder.analyze_image(args.image_path)

        # Draw elements on the image
        ui_finder.draw_elements(args.image_path, ui_elements, args.output)
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
