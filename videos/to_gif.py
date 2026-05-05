from moviepy import VideoFileClip

def mp4_to_gif(input_path, output_path, fps=10, start=0, end=None):
    clip = VideoFileClip(input_path)
    
    if end:
        clip = clip.subclip(start, end)
    
    clip.write_gif(output_path, fps=fps)

mp4_to_gif("run4_sparse.mp4", "run4_sparse.gif", fps=10)#, start=0, end=5)