#!/usr/bin/env python3
#
# For any tool that reads binary packets of size <N> and writes
# results as binary packets of size <M>, this tool will parallelize
# it: it will read input from stdin <N> bytes at a time, split it
# into multiple files, call the specified program in parallel,
# merge the resulting files, and print the output.
#
# Usage:
#    partransform.py <N> <M> /path/to/program program-args ...

import os
import subprocess
import sys
import tempfile

def read_in_packets(f, size):
    while True:
        packet = f.read(size)
        if len(packet) == 0: break
        if len(packet) != size: raise IOError(f"got only {len(packet)} bytes of {size} expected")
        yield packet

def partransform(command, input_packets, output_packet_size, nworkers=None):
    if nworkers is None:
        try:
            nworkers = max(1, len(os.sched_getaffinity(0)))
        except AttributeError:
            nworkers = max(1, os.cpu_count())

    files = [
        tempfile.NamedTemporaryFile(prefix="parevalB.")
        for i in range(nworkers)
    ]

    workers = [
        subprocess.Popen(command, stdin=subprocess.PIPE, stdout=files[i])
        for i in range(nworkers)
    ]

    packet_to_worker_id = []
    worker_packets_sent = [0] * nworkers
    for i, packet in enumerate(input_packets):
        idx = i % nworkers
        workers[idx].stdin.write(packet)
        packet_to_worker_id.append(idx)
        worker_packets_sent[idx] += 1

    worker_packets = []
    worker_packets_done = []
    for i in range(nworkers):
        workers[i].stdin.flush()
        workers[i].stdin.close()
    for i in range(nworkers):
        workers[i].wait()
        workers[i] = None
        with open(files[i].name, "rb") as f:
            worker_packets.append(list(read_in_packets(f, output_packet_size)))
        worker_packets_done.append(0)
        if len(worker_packets[i]) != worker_packets_sent[i]:
            raise IOError(f"worker {i} returned {len(worker_packets[i])} out of {worker_packets_sent[i]} required packets")
        files[i].close()
        files[i] = None

    for i in range(len(packet_to_worker_id)):
        wid = packet_to_worker_id[i]
        packet = worker_packets[wid][worker_packets_done[wid]]
        worker_packets_done[wid] += 1
        yield packet

if __name__ == "__main__":

    insize = int(sys.argv[1])
    outsize = int(sys.argv[2])
    command = sys.argv[3:]

    input_packets = read_in_packets(sys.stdin.buffer, insize)

    for output_packet in partransform(command, input_packets, outsize):
        sys.stdout.buffer.write(output_packet)
