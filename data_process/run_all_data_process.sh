#!/usr/bin/env bash
echo -e "\033[44;37;5m RUNNING embed2ndarray.py\033[0m ";
python embed2ndarray.py;
echo -e "\033[44;37;5m RUNNING question_and_topic_2id.py\033[0m ";
python question_and_topic_2id.py;
echo -e "\033[44;37;5m RUNNING char2id.py\033[0m ";
python char2id.py;
echo -e "\033[44;37;5m RUNNING word2id.py\033[0m ";
python word2id.py;
echo -e "\033[44;37;5m RUNNING creat_batch_data.py\033[0m ";
python creat_batch_data.py;
echo -e "\033[44;37;5m RUNNING creat_batch_seg.py\033[0m ";
python creat_batch_seg.py;