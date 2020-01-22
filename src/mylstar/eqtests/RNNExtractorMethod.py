# -*- coding: utf-8 -*-

# Derived from :
# +---------------------------------------------------------------------------+
# | pylstar : Implementation of the LSTAR Grammatical Inference Algorithm     |
# +---------------------------------------------------------------------------+
# | Copyright (C) 2015 Georges Bossert                                        |
# | This program is free software: you can redistribute it and/or modify      |
# | it under the terms of the GNU General Public License as published by      |
# | the Free Software Foundation, either version 3 of the License, or         |
# | (at your option) any later version.                                       |
# |                                                                           |
# | This program is distributed in the hope that it will be useful,           |
# | but WITHOUT ANY WARRANTY; without even the implied warranty of            |
# | MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the              |
# | GNU General Public License for more details.                              |
# |                                                                           |
# | You should have received a copy of the GNU General Public License         |
# | along with this program. If not, see <http://www.gnu.org/licenses/>.      |
# +---------------------------------------------------------------------------+
# | @url      : https://github.com/gbossert/pylstar                           |
# | @contact  : gbossert@miskin.fr                                            |
# +---------------------------------------------------------------------------+



# +----------------------------------------------------------------------------
# | Global Imports
# +----------------------------------------------------------------------------
import random

# +----------------------------------------------------------------------------
# | pylstar Imports
# +----------------------------------------------------------------------------
from mylstar.tools.Decorators import PylstarLogger
from mylstar.Word import Word
from mylstar.Letter import EmptyLetter
from mylstar.OutputQuery import OutputQuery
from mylstar.eqtests.RNNQuantisations import SVMDecisionTreeQuantisation
from mylstar.eqtests.RNNCounterexampleGenerator import WhiteboxRNNCounterexampleGenerator


@PylstarLogger
class RNNExtractorMethod(object):
    """This algorithm is only usefule when extracting an automata from a trained
    recurrent neural network. 

    It is based on the lstar_extractor algorithm described in (cite ICML18 paper).

    The algorithm uses clustered values of hidden states to inform the creation
    of counter examples. 

    XXXjc Add doctests eventually. 

    """
    
    def __init__(self, knowledge_base, input_letters, network, max_steps, 
                    num_dims_initial_split=10, starting_examples=None):
        self.knowledge_base = knowledge_base
        self.input_letters = input_letters
        self.max_steps = max_steps
        self.network = network
        self.discretiser = SVMDecisionTreeQuantisation(num_dims_initial_split)
        self.counterexample_generator = WhiteboxRNNCounterexampleGenerator(network,self,self.discretiser,starting_examples)


    def find_counterexample(self, hypothesis):
        if hypothesis is None:
            raise Exception("Hypothesis cannot be None")

        self._logger.info("Starting the to search for a WhiteboxRNN counter-example")
        cex,msg = self.counterexample_generator.counterexample(hypothesis)
        print(msg)
        query = OutputQuery(cex)
        self.knowledge_base.resolve_query(query)
        print(query)
        return query



        #query = OutputQuery(input_word)
        #self.knowledge_base.resolve_query(query)

        #if query.output_word != expected_output_word:
        #    self._logger.info("Found a counter-example : input: '{}', expected: '{}', observed: '{}'".format(input_word, expected_output_word, query.output_word))
        #    return query
        #return None

    def minimal_diverging_suffix(self,state1,state2): #gets series of letters showing the two states are different,
        # i.e., from which one state reaches accepting state and the other reaches rejecting state
        # assumes of course that the states are in the automaton and actually not equivalent
        res = None
        # just use BFS til you reach an accepting state
        # after experiments: attempting to use symmetric difference on copies with s1,s2 as the starting state, or even
        # just make and minimise copies of this automaton starting from s1 and s2 before starting the BFS,
        # is slower than this basic BFS, so don't
        seen_states = set()
        new_states = {(Word(),(state1,state2))}
        while len(new_states) > 0:
            prefix,state_pair = new_states.pop()
            s1,s2 = state_pair
            if s1 != s2: # intersection of self.F and [s1,s2] is exactly one state,
                # meaning s1 and s2 are classified differently
                res = prefix
                break
            seen_states.add(state_pair)
            for a in self.input_letters:
                next_state_pair = (s1.visit(a)[1],s2.visit(a)[1])
                next_tuple = (prefix.letters.append(a),next_state_pair)
                if not next_tuple in new_states and not next_state_pair in seen_states:
                    new_states.add(next_tuple)
        return res
