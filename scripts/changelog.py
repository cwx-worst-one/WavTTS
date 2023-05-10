#!/usr/bin/env python3
"""This is changelog module

    this module is designed to make a md file of the commits which are merged into \
    the develop branch of dolphin between the begin commit and the end commit.\
    you can use this module to generate your changelog md by provide three parameters.\
    Use "--begin_commit 257bd0c3(or SHA)" to set the begin commit.\
    Use "--end_commit f891cb9a(or SHA)" to set the end commit.\
    Use "--version version" to set the version of dolphin.

"""
import os
from datetime import datetime
from argparse import ArgumentParser

from git.repo import Repo
from git.repo.fun import is_git_dir


ALLOWED_TYPE = ["feat", "fix", "docs", "style", "refactor", "test", "chore"]


class GitRepository(object):
    """git仓库管理"""
    def __init__(self, local_path, repo_url, branch='master'):
        self.local_path = local_path
        self.repo_url = repo_url
        self.repo = None
        self.log_list_indict=[]
        self.init_repo(repo_url, branch)

    def init_repo(self, repo_url, branch):
        """初始化git仓库"""
        if not os.path.exists(self.local_path):
            os.makedirs(self.local_path)

        git_local_path = os.path.join(self.local_path, '.git')
        if not is_git_dir(git_local_path):
            self.repo = Repo.clone_from(repo_url, to_path=self.local_path, branch=branch)
        else:
            self.repo = Repo(self.local_path)

    def pull(self):
        """从线上拉最新代码"""
        self.repo.git.pull()

    def collect_commit_logs(self, commit_begin=None, commit_end=None):
        """获取所有提交记录

        commit_begin: 不包含；
        commit_end: 包含；
        """
        log_list_indict=[]
        commit_log = self.repo.git.log(
            '--pretty={"commit":"%h","author":"%an","summary":"%s","body":"%b","date":"%cd"}',
            '--merges',
            commit_begin + "..." + commit_end,
            date='format:%Y-%m-%d %H:%M')
        commit_log = commit_log.replace('\n\n', '')
        commit_logs = commit_log.split("\n")
        for commit in commit_logs:
            try:
                log_list_indict.append(eval(commit))
            except:
                print('出现错误' + commit)
        self.log_list_indict=log_list_indict
        return log_list_indict

    def checkout(self, branch):
        """切换分支"""
        self.repo.git.checkout(branch)

    def parse_commit(self, loglist=None):
        """提取log中的有效信息"""
        commitdict = {}
        for log in loglist:
            body = log["body"]
            if body != '' or None:
                body_split = body.split(sep = ':')
                type_by_scope = body_split[0]
                try:
                    summary_by_link = body_split[1]
                    commitdict.setdefault(type_by_scope,[]).append(summary_by_link)
                except:
                    print("")
        return commitdict

def generate_changelog(local_path, loglist=None, pretype=None, version='latest'):
    history_changelog_list=[]
    joined_path = os.path.join(local_path,'CHANGELOG.md')
    history_file=open(joined_path, "r")
    for line in history_file:
        history_changelog_list.append(line)
    history_file.close()
    cur_time = datetime.now()
    date = (cur_time.strftime("%Y-%m-%d"))
    f = open(joined_path, 'w')
    f.write("### " + version + ' ' + date + "\n")
    for line in loglist:
        type_and_scope = line[0]
        summary_and_link_list = line[1]
        type_and_scope_splited = type_and_scope.split(sep='(')
        commit_type = type_and_scope_splited[0]
        try:
            scope = type_and_scope_splited[1][:-1]
        except:
            scope = 'rest'
        if commit_type not in ALLOWED_TYPE:
            continue
        if commit_type != pretype:
            pretype = commit_type
            f.write("\n#### " + commit_type + "\n\n")
        f.write("* **" + scope + ":**" + '\n')
        for summary_and_link in summary_and_link_list:
            summary_and_link_split = summary_and_link.split(sep='See merge request lab-speech/dolphin!')
            summary = summary_and_link_split[0]
            link = summary_and_link_split[1]
            f.write(f'  *  {summary}[(MR{link})](https://code.byted.org/lab-speech/dolphin/merge_requests/{link})\n')
    f.write('\n---\n\n')
    for line in history_changelog_list:
        f.write(line)
    f.close()


def parse_args():
    '''parse args'''
    parser = ArgumentParser(description='changelog parser')
    parser.add_argument('--begin_commit', default="9325d096", help='not include')
    parser.add_argument('--end_commit', default="5197f961", help='included')
    parser.add_argument('--version', default="latest", help='where the changelog end')
    return parser.parse_known_args()


def main():
    path = os.path.abspath(__file__)
    local_path = os.path.dirname(os.path.dirname(path))
    remote_path = "https://code.byted.org/lab-speech/dolphin.git"
    repo = GitRepository(local_path, remote_path)
    repo.checkout('develop')
    repo.pull()
    args, unknown = parse_args()
    repo.collect_commit_logs(args.begin_commit, args.end_commit)
    commit_dict = repo.parse_commit(repo.log_list_indict)
    commit_dict_sorted = sorted(commit_dict.items(), key=lambda d:d[0])
    generate_changelog(local_path, commit_dict_sorted, None, args.version)


if __name__ == '__main__':
    main()
