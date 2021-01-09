from python:3.8.7-slim-buster
#  Test & CICD harness for the code
# docker build -t mlyoucanusetest .
# docker run mlyoucanusetest:latest
LABEL maintainer="todd.g.cook@gmail.com"
# get & apply security updates
RUN apt-get update && apt-get install -y apt-utils \
	&& apt-get -s dist-upgrade | grep "^Inst" | \
	grep -i securi | awk -F " " {'print $2'} | \
	xargs apt-get install -y --fix-missing \
	&&  pip install --upgrade pip \
	&&  useradd -ms /bin/bash user
USER user
WORKDIR /home/user
ENV PATH /home/user/.local/bin:/bin:/usr/local/bin:/usr/bin
COPY requirements.txt .
RUN pip install --user -r requirements.txt
COPY install_corpora.py install_corpora.py
# RUN python3 install_corpora.py
# RUN python3 -m nltk.downloader all
COPY mlyoucanuse mlyoucanuse
COPY runUnitTests.sh runUnitTests.sh
COPY .pylintrc .pylintrc
COPY .coveragerc .coveragerc
COPY mypy.ini mypy.ini
ENV PYTHONPATH .:/user/.local/bin
CMD ["/bin/bash", "runUnitTests.sh"]