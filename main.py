import argparse
from scripts.general import preprocess_image, preprocess_video, postprocess_predictions, save_video_with_predictions, draw_predictions
from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput



def infer_onnx(input_data, model_name="yolo_model", server_url="localhost:8000"):
    """Run inference using Triton server."""
    # from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput
    client = InferenceServerClient(url=server_url)

    # Match input name in config.txt
    inputs = InferInput("images", input_data.shape, "FP32")
    inputs.set_data_from_numpy(input_data)

    # Match output name in config.txt
    outputs = InferRequestedOutput("output0")

    # Send request to Triton
    response = client.infer(model_name, inputs=[inputs], outputs=[outputs])
    return response.as_numpy("output0")



def main():
    parser = argparse.ArgumentParser(description="Run inference with Triton server")
    parser.add_argument("--input", type=str, required=True, help="Path to image or video")
    parser.add_argument("--output", type=str, default="output.mp4", help="Path to save output (for video inputs)")
    parser.add_argument("--server_url", type=str, default="localhost:8000", help="Triton server URL")
    args = parser.parse_args()

    if args.input.lower().endswith((".jpg", ".jpeg", ".png")):
        # Process image
        preprocessed_input, original_shape = preprocess_image(args.input)
        output = infer_onnx(preprocessed_input, server_url=args.server_url)
        predictions = postprocess_predictions(output, original_shape)
        print("Predictions:", predictions)

        # Draw predictions and save the output image
        draw_predictions(args.input, predictions, output_path="output_image.jpg")
        print("Predictions drawn on image and saved to output.jpg")
    elif args.input.lower().endswith((".mp4", ".avi", ".mov")):
        # Process video
        frame_generator, total_frames, original_shape = preprocess_video(args.input)
        predictions_per_frame = []

        for frame, preprocessed_frame in frame_generator:
            output = infer_onnx(preprocessed_frame, server_url=args.server_url)
            predictions = postprocess_predictions(output, original_shape)
            predictions_per_frame.append((frame, predictions))

        save_video_with_predictions(args.input, args.output, predictions_per_frame)
        print(f"Processed video saved at {args.output}")
    else:
        print("Unsupported file type. Please provide an image or video file.")


if __name__ == "__main__":
    main()
