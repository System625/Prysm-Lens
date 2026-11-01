from service.gradio_app import demo

demo.queue(max_size=32).launch(max_file_size="200mb")
