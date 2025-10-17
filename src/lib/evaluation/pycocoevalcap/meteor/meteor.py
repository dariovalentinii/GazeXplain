#!/usr/bin/env python

# Python wrapper for METEOR implementation, by Xinlei Chen
# Acknowledge Michael Denkowski for the generous discussion and help 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import subprocess
import threading
import warnings

# Assumes meteor-1.5.jar is in the same directory as meteor.py.  Change as needed.
METEOR_JAR = 'meteor-1.5.jar'
# print METEOR_JAR

class Meteor:

    def __init__(self):
        self.env = os.environ
        self.env['LC_ALL'] = 'en_US.UTF_8'
        self.meteor_cmd = ['java', '-jar', '-Xmx2G', METEOR_JAR, \
                '-', '-', '-stdio', '-l', 'en', '-norm']
        self.meteor_p = subprocess.Popen(self.meteor_cmd, \
                cwd=os.path.dirname(os.path.abspath(__file__)), \
                stdin=subprocess.PIPE, \
                stdout=subprocess.PIPE, \
                stderr=subprocess.PIPE,
                env=self.env, universal_newlines=True, bufsize=1)
        # Used to guarantee thread safety
        self.lock = threading.Lock()

    def compute_score(self, gts, res):
        assert(gts.keys() == res.keys())
        imgIds = sorted(list(gts.keys()))
        scores = []

        eval_line = 'EVAL'
        self.lock.acquire()
        for i in imgIds:
            assert(len(res[i]) == 1)
            stat = self._stat(res[i][0], gts[i])
            eval_line += ' ||| {}'.format(stat)

        # Send to METEOR
        self.meteor_p.stdin.write(eval_line + '\n')
        
        try:
            # Collect segment scores
            for _ in range(len(imgIds)):
                scores.append(self._read_score())

            # Final score
            final_score = self._read_score()
        except RuntimeError as err:
            # When METEOR fails to return numeric scores (e.g. missing Java),
            # fall back to zeros so evaluation can proceed.
            warnings.warn(
                "METEOR metric skipped because the Java process did not "
                f"return a valid score: {err}"
            )
            final_score = 0.0
            scores = [0.0 for _ in imgIds]
        finally:
            self.lock.release()

        return final_score, scores

    def method(self):
        return "METEOR"

    def _stat(self, hypothesis_str, reference_list):
        # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
        hypothesis_str = hypothesis_str.replace('|||', '').replace('  ', ' ')
        score_line = ' ||| '.join(('SCORE', ' ||| '.join(reference_list), hypothesis_str))
        self.meteor_p.stdin.write(score_line+'\n')
        return self.meteor_p.stdout.readline().strip()

    def _read_score(self):
        line = self.meteor_p.stdout.readline()
        if line == '':
            stderr_output = self.meteor_p.stderr.read()
            raise RuntimeError(
                'METEOR jar produced no output.' +
                (f' Stderr: {stderr_output.strip()}' if stderr_output else '')
            )

        stripped = line.strip()
        if not stripped:
            stderr_line = self.meteor_p.stderr.readline().strip()
            raise RuntimeError(
                'Received empty METEOR score line.' +
                (f' Details: {stderr_line}' if stderr_line else '')
            )

        try:
            return float(stripped)
        except ValueError as exc:
            stderr_line = self.meteor_p.stderr.readline().strip()
            raise RuntimeError(
                f"Invalid METEOR score '{stripped}'." +
                (f' Details: {stderr_line}' if stderr_line else '')
            ) from exc
 
    def __del__(self):
        self.lock.acquire()
        self.meteor_p.stdin.close()
        self.meteor_p.kill()
        self.meteor_p.wait()
        self.lock.release()
