FROM python:3.8

MAINTAINER "chatGLM"

COPY requirements.txt /chatGLM/
WORKDIR /chatGLM

RUN pip install --user torch torchvision tensorboard cython -i https://pypi.tuna.tsinghua.edu.cn/simple
# RUN pip install --user 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

# RUN pip install --user 'git+https://github.com/facebookresearch/fvcore'
# install detectron2
# RUN git clone https://github.com/facebookresearch/detectron2

RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/ --trusted-host pypi.tuna.tsinghua.edu.cn

CMD ["python","-u", "webui.py"]
