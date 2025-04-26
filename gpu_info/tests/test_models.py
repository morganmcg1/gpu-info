from gpu_info.models.models import VideoInfo, DetailedContent

# Create a VideoInfo instance
video_info = VideoInfo(
    video_id="D7_ipDqhtwk",
    title="Test Video",
    author="Test Author",
    length=3600,
    url="https://www.youtube.com/watch?v=D7_ipDqhtwk",
    description="Test Description",
    transcript="Test Transcript",
    thumbnail_url="https://example.com/thumbnail.jpg",
    publish_date="2023-01-01",
    views=1000,
    metadata={"key": "value"}
)

# Create a DetailedContent instance
detailed_content = DetailedContent(
    video_url="https://www.youtube.com/watch?v=D7_ipDqhtwk",
    detailed_content="Test Detailed Content",
    is_cuda_content=True,
    content_type="tutorial",
    technical_level="intermediate",
    key_topics=["CUDA", "GPU", "Parallel Computing"]
)

print(f"VideoInfo: {video_info}")
print(f"DetailedContent: {detailed_content}")
