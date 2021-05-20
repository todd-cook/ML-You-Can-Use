from python:3.8.7-slim-buster
# Test & CICD harness for the code
# docker build -t mlyoucanusetest .
# docker run test_mlyoucanuse:latest
# or # docker run -it --rm test_mlyoucanuse ./runUnitTests.sh
LABEL maintainer="https://github.com/todd-cook"
# get & apply security updates
RUN apt-get update && apt-get install -y apt-utils g++ \
	&& apt-get -s dist-upgrade | grep "^Inst" | \
	grep -i securi | awk -F " " {'print $2'} | \
	xargs apt-get install -y --fix-missing \
	&&  pip install --upgrade pip
RUN groupadd -r user && \
    useradd -d /home/user -r -s /bin/bash -g user user && \
    chown -R user:user /home/user || :
USER user
WORKDIR /home/user
ENV PATH /home/user/.local/bin:/bin:/usr/local/bin:/usr/bin
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt
COPY mlyoucanuse mlyoucanuse
COPY install_corpora.py runUnitTests.sh .pylintrc .coveragerc mypy.ini /home/user/
# RUN python3 install_corpora.py
# RUN python3 -m nltk.downloader all
USER root
RUN chown -R user:user /home/user
RUN chmod ug+x /home/user/runUnitTests.sh
USER user
ENV PYTHONPATH .:/user/.local/bin
CMD ["/bin/bash", "runUnitTests.sh"]