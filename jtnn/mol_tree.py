import rdkit
import rdkit.Chem as Chem
import copy
from .chemutils import get_clique_mol, tree_decomp, get_mol, get_smiles, set_atommap, enum_assemble, decode_stereo

def get_slots(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return [(atom.GetSymbol(), atom.GetFormalCharge(), atom.GetTotalNumHs()) for atom in mol.GetAtoms()]

class Vocab(object):

    def __init__(self, smiles_list):
        self.vocab = smiles_list
        self.vmap = {x:i for i,x in enumerate(self.vocab)}
        self.slots = [get_slots(smiles) for smiles in self.vocab]
        
    def get_index(self, smiles):
        return self.vmap[smiles]

    def get_smiles(self, idx):
        return self.vocab[idx]

    def get_slots(self, idx):
        return copy.deepcopy(self.slots[idx])

    def size(self):
        return len(self.vocab)

class MolTreeNode(object):

    def __init__(self, smiles, clique=[]):
        self.smiles = smiles
        self.mol = get_mol(self.smiles)                                        # node.mol

        self.clique = [x for x in clique] #copy
        self.neighbors = []
        
    def add_neighbor(self, nei_node):
        self.neighbors.append(nei_node)                                        #node.neighbor

    def recover(self, original_mol):
        clique = []
        clique.extend(self.clique)                                                  #clique of substructure
        if not self.is_leaf:
            for cidx in self.clique:
                original_mol.GetAtomWithIdx(cidx).SetAtomMapNum(self.nid)           #set the substructure's id in line 91 to the main structure


        for nei_node in self.neighbors:                                             #neighbors contain node neighbor
            clique.extend(nei_node.clique)
            if nei_node.is_leaf: #Leaf node, no need to mark 
                continue
            for cidx in nei_node.clique:                                                #call node.clique is the clique in line 12
                #allow singleton node override the atom mapping                     #A singleton node refers to a node in the clique that forms a clique by itself (contains only one atom).    
                if cidx not in self.clique or len(nei_node.clique) == 1:            #check idx in clique of neighbor node not in the current node or just a singleton         
                    atom = original_mol.GetAtomWithIdx(cidx)
                    atom.SetAtomMapNum(nei_node.nid)

        clique = list(set(clique))
        label_mol = get_clique_mol(original_mol, clique)
        self.label = Chem.MolToSmiles(Chem.MolFromSmiles(get_smiles(label_mol)))
        self.label_mol = get_mol(self.label)

        for cidx in clique:
            original_mol.GetAtomWithIdx(cidx).SetAtomMapNum(0)

        return self.label                                                           #mol of clique contain index        
    
    def assemble(self):
        neighbors = [nei for nei in self.neighbors if nei.mol.GetNumAtoms() > 1]                #neigh not singleton
        neighbors = sorted(neighbors, key=lambda x:x.mol.GetNumAtoms(), reverse=True)           #extracts the number of atoms in each molecule, and reverse=True sorts the molecules in descending
        singletons = [nei for nei in self.neighbors if nei.mol.GetNumAtoms() == 1]
        neighbors = singletons + neighbors

        cands,aroma = enum_assemble(self, neighbors)
        new_cands = [cand for i,cand in enumerate(cands) if aroma[i] >= 0]
        if len(new_cands) > 0: cands = new_cands

        if len(cands) > 0:
            self.cands, _ = zip(*cands)
            self.cands = list(self.cands)
        else:
            self.cands = []
            
class MolTree(object):

    def __init__(self, smiles):
        self.smiles = smiles
        self.mol = get_mol(smiles)

        #Stereo Generation
        mol = Chem.MolFromSmiles(smiles)
        self.smiles3D = Chem.MolToSmiles(mol, isomericSmiles=True)
        self.smiles2D = Chem.MolToSmiles(mol)
        self.stereo_cands = decode_stereo(self.smiles2D)

        cliques, edges = tree_decomp(self.mol)                              #tree_decompose in chemutils.py
        self.nodes = []
        root = 0
        for i,c in enumerate(cliques):
            cmol = get_clique_mol(self.mol, c)                              #get the mol of each clique
            node = MolTreeNode(get_smiles(cmol), c)
            self.nodes.append(node)
            if min(c) == 0:                                                 #finding root node by find clique contain atom has index 0 
                root = i

        for x,y in edges:                                                   #edge contain index of clique that connected
            self.nodes[x].add_neighbor(self.nodes[y])
            self.nodes[y].add_neighbor(self.nodes[x])
        
        if root > 0:                                                        #ensuring that the identified root node is positioned at index 0 in the self.nodes
            self.nodes[0],self.nodes[root] = self.nodes[root],self.nodes[0]

        for i,node in enumerate(self.nodes):
            node.nid = i + 1
            if len(node.neighbors) > 1: #Leaf node mol is not marked
                set_atommap(node.mol, node.nid)                             #Set the ID of each node (the mol of each clique), no neighbor = leaf
            node.is_leaf = (len(node.neighbors) == 1)

    def size(self):
        return len(self.nodes)

    def recover(self):
        for node in self.nodes:
            node.recover(self.mol)                                          #recover in Mol_Tree Node

    def assemble(self):
        for node in self.nodes:
            node.assemble()

if __name__ == "__main__":
    import sys
    lg = rdkit.RDLogger.logger() 
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    cset = set()
    for i,line in enumerate(sys.stdin):
        smiles = line.split()[0]
        mol = MolTree(smiles)
        for c in mol.nodes:
            cset.add(c.smiles)
    for x in cset:
        print(x)

