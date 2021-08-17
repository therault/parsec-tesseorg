# parsec_dag.py
# Python module to represent a DAG of tasks for PaRSEC profiling

import networkx as nx
from collections import namedtuple
import re
import sys

REQUIRED_NX_VERSION = [2, 0]


class ParsecDAG:
    """A DAG of PaRSEC tasks"""

    def __init__(self):
        if self.cmp([int(x) for x in re.sub(r'(\.0+)*$', '', nx.__version__).split(".")], REQUIRED_NX_VERSION) < 0:
            raise Exception("NetworkX '{}' or more is required, version '{}' is available".format(
                             ".".join(str(x) for x in REQUIRED_NX_VERSION), nx.__version__))
        self.dag = nx.DiGraph()
        self.idtoname = dict()
        self.nametoid = dict()
        self.ParsecTaskID = namedtuple("ParsecTaskID", ["tpid", "tcid", "tid"])

    @staticmethod
    def cmp(a, b):
        return (a > b) - (a < b)

    def _load_parsec_dot_file(self, f):
        """Module-private function. Adds nodes and links found in a file to the DAG being built

        Parameters
        ----------
        f: str
           A file name (DOT file, generated by PaRSEC with PARSEC_PROF_GRAPHER)

        Returns
        -------
        Nothing, but add nodes and links to the DAG

        Raises
        ------
        All execeptions due to file errors
        Additional exceptions when the file does not follow the format that PaRSEC is supposed to produce
        """
        node = re.compile(r'''
           (?P<name>[^ ]+)                       #Group name is all the characters to the first space
           .*label=".(?P<thid>[0-9]+)            #Group thid is the first integer in "<a/b>" at the begining of the label
           .(?P<vpid>[0-9]+)                     #Group vpid is the second integer in "<a/b>" at the begining of the label
           [^a-zA-Z_]*                           #Skip until the first letter
           (?P<label>[^(]+)                      #Group label is everything until the '(' 
           .(?P<param>[^)]+)                     #Group param follows the '(', it's all the things until ')' 
           .(?P<local>[^<]+)                     #Group local follows the ')', it's all the things until '<' 
           .(?P<prio>[0-9]+)                     #Group prio is the inside of <prio>
           [^{]*                                 #Skip until the '{'
           .(?P<tpid>[0-9]+)                     #Group tpid is the inside of {tpid}
           .*tpid=(?P<tt_tpid>[0-9]+)            #Skip until tpid=, and read group tt_tpid
           .*tcid=(?P<tt_tcid>[0-9]+)            #Skip until tcid=, and read group tt_tcid
           .*tcname=(?P<tt_tcname>[^:]+)         #Skip until tcname=, and read group tt_tcname
           .*tid=(?P<tt_tid>[0-9]+)              #Skip until tid=, and read group tt_tid''', re.VERBOSE)
        link = re.compile('''
           (?P<src>[^ ]+)                        #Group src is everything to the first space
           [^a-zA-Z0-9_]*(?P<dst>[^ ]+)          #Group dst is everything alphanumeric after that, to the first space
           .*label="(?P<flow_src>[^=]+)          #Group flow_src is the first thing before '=' after label="
           =.(?P<flow_dst>[^,]+)                 #Group flow_dst is everything to ',' after =>
           .*color="(?P<color>[^"]+)             #Group color is everything inside color="..."
           .*style="(?P<style>[^"]+)             #Group style is everything inside style="..." ''', re.VERBOSE)
        start = re.compile('digraph G {')
        end = re.compile('}')
        nb = 1
        with open(f) as fp:
            line = fp.readline()
            while line:
                res = node.match(line)
                if res:
                    if len(res.groups()) != 12:
                        estr = "Node lines are expected to provide 12 arguments, {} found in `{}` (line {} of {})".format(len(res.groups()), line, nb, f)
                        raise Exception(estr)
                    if int(res.group('tt_tpid')) != int(res.group('tpid')):
                        estr = 'Node `{}` at line {} has inconsistent taskpool ids {} and {}'.format(line, nb, int(res.group('tpid')), int(res.group('tt_tpid')))
                        raise Exception()
                    name = res.group('name')
                    parsec_id = self.ParsecTaskID(tpid=int(res.group('tt_tpid')),
                                                  tid=int(res.group('tt_tid')),
                                                  tcid=int(res.group('tt_tcid')))
                    self.idtoname[parsec_id] = name
                    self.nametoid[name] = parsec_id
                    self.dag.add_node(name, thid=int(res.group('thid')), vpid=int(res.group('vpid')),
                                      label=res.group('label'), param=res.group('param'), local=res.group('local'),
                                      prio=int(res.group('prio')), tcid=int(res.group('tt_tcid')), tid=int(res.group('tt_tid')),
                                      tpid=int(res.group('tt_tpid')))
                else:
                    res = link.match(line)
                    if res:
                        if len(res.groups()) != 6:
                            raise Exception('Link lines are expected to provide 6 arguments, {} found in `{}` (line {} of {})' .format(
                                             len(res.groups()), line, nb, f))
                        src = res.group('src')
                        dst = res.group('dst')
                        self.dag.add_edge(src, dst, flow_src=res.group('flow_src'),
                                          flow_dst=res.group('flow_dst'), color=res.group('color'),
                                          style=res.group('style'))
                    else:
                        res = start.match(line)
                        if not res:
                            res = end.match(line)
                            if not res:
                                raise Exception('Line `{}` does not match node or link (line {} of {})'.format(line, nb, f))
                line = fp.readline()
                nb += 1

    def load_parsec_dot_files(self, files):
        """Builds a NetworkX DiGraph from a set of DOT files generated by PaRSEC Prof Grapher

        Parameters
        ----------
        files: list of str
          The files to load

        Returns
        -------
        Nothing, but loads the files into the internal dag

        dag: networkx.DiGraph
          a Directed Graph holding all the nodes (tasks) and edges (flows) found in the dot files
          Each node is decorated with thid (thread id), vpid (virtual process id), label (task class name),
            param (parameters), local (values of local variables), prio (priority of the task),
            tcid (task class ID, corresponds to the 'type' of events in the HDF5), tid (Task Identifier,
            corresponds to the 'id' of events in the HDF5, and tpid (Taskpool Identifier, corresponds to
            the 'taskpool_id' of events in the HDF5).
          Each edge is decorated with flow_src (the name of the source flow), flow_dst (the name of the
            destination flow), color (a suggested color for the link), and style (a suggested style for
            the link).
        """
        for f in files:
            self._load_parsec_dot_file(f)

    def node_from_name(self, name):
        """Returns a node from its internal name"""
        return self.dag.nodes[name]

    def node_from_id(self, tpid, tcid, tid):
        """Returns a node from its identifiers"""
        parsec_id = self.ParsecTaskID(tpid=int(tpid), tcid=int(tcid), tid=int(tid))
        return self.dag.nodes[self.idtoname[parsec_id]]

    def nodename_from_id(self, tpid, tcid, tid):
        """Returns the internal name of a node from its identifiers"""
        parsec_id = self.ParsecTaskID(tpid=int(tpid), tcid=int(tcid), tid=int(tid))
        return self.idtoname[parsec_id]

    def successors_from_name(self, name):
        """Returns the list of nodes successors of name"""
        return self.dag[name]

    def successors_from_id(self, tpid, tcid, tid):
        """Returns the list of nodes successors of (tpid, tcid, tid)"""
        parsec_id = self.ParsecTaskID(tpid=int(tpid), tcid=int(tcid), tid=int(tid))
        return self.dag[self.idtoname[parsec_id]]


if __name__ == '__main__':
    print("Loading all DOT files: '{}'".format(",".join(sys.argv[1:])))
    dag = ParsecDAG()
    dag.load_parsec_dot_files(sys.argv[1:])
    print("{}".format(type(dag)))
    print("DAG has {} nodes and {} edges".format(nx.number_of_nodes(dag.dag), nx.number_of_edges(dag.dag)))
