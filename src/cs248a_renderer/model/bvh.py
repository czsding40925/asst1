import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Callable
import numpy as np
import slangpy as spy
from torch import threshold

from cs248a_renderer.model.bounding_box import BoundingBox3D
from cs248a_renderer.model.primitive import Primitive
from tqdm import tqdm


logger = logging.getLogger(__name__)

from collections import deque
from dataclasses import dataclass
from typing import Deque, Tuple

@dataclass
class WorkItem:
    node_idx: int
    start: int
    end: int
    depth: int
    
@dataclass
class BVHNode:
    # The bounding box of this node.
    bound: BoundingBox3D = field(default_factory=BoundingBox3D)
    # The index of the left child node, or -1 if this is a leaf node.
    left: int = -1
    # The index of the right child node, or -1 if this is a leaf node.
    right: int = -1
    # The starting index of the primitives in the primitives array.
    prim_left: int = 0
    # The ending index (exclusive) of the primitives in the primitives array.
    prim_right: int = 0
    # The depth of this node in the BVH tree.
    depth: int = 0

    def get_this(self) -> Dict:
        return {
            "bound": self.bound.get_this(),
            "left": self.left,
            "right": self.right,
            "primLeft": self.prim_left,
            "primRight": self.prim_right,
            "depth": self.depth,
        }

    @property
    def is_leaf(self) -> bool:
        """Checks if this node is a leaf node."""
        return self.left == -1 and self.right == -1


class BVH:
    def __init__(
        self,
        primitives: List[Primitive],
        max_nodes: int,
        min_prim_per_node: int = 1,
        num_thresholds: int = 16,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> None:
        """
        Builds the BVH from the given list of primitives. The build algorithm should
        reorder the primitives in-place to align with the BVH node structure.
        The algorithm will start from the root node and recursively partition the primitives
        into child nodes until the maximum number of nodes is reached or the primitives
        cannot be further subdivided.
        At each node, the splitting axis and threshold should be chosen using the Surface Area Heuristic (SAH)
        to minimize the expected cost of traversing the BVH during ray intersection tests.

        :param primitives: the list of primitives to build the BVH from
        :type primitives: List[Primitive]
        :param max_nodes: the maximum number of nodes in the BVH
        :type max_nodes: int
        :param min_prim_per_node: the minimum number of primitives per leaf node
        :type min_prim_per_node: int
        :param num_thresholds: the number of thresholds per axis to consider when splitting
        :type num_thresholds: int
        """
        self.nodes: List[BVHNode] = []
        self.primitives = primitives
        self.max_nodes = max_nodes
        self.min_prim_per_node = min_prim_per_node
        self.num_thresholds = num_thresholds
        self.on_progress = on_progress 
        # self.build_node(0, len(primitives), 0)
        self.balance_min_frac = 0.05
        self.max_depth = 10
        self.build_bfs()


    # TODO: Student implementation starts here.

    def SAH_split(self, start, end):
        n = end - start 
        if n <= 1: 
            return None 

        # get the centers of the primitives 
        centers = [self.primitives[i].bounding_box.center for i in range(start, end)] 
        cmin = [min(c[a] for c in centers) for a in range(3)]
        cmax = [max(c[a] for c in centers) for a in range(3)]

        best = None
        best_cost = float("inf")

        for axis in range(3):
            low, high = cmin[axis], cmax[axis]
            if high <= low:
                continue 

            for k in range(self.num_thresholds): 
                
                thresh = low + (k + 1) / (self.num_thresholds + 1) * (high - low)
                NL, NR = 0, 0 
                BL, BR = None, None

                for i in range(start, end): 
                    p = self.primitives[i]
                    if p.bounding_box.center[axis] <= thresh:
                        NL += 1 
                        BL = p.bounding_box if BL is None else BoundingBox3D.union(BL, p.bounding_box) 
                    else: 
                        NR += 1
                        BR = p.bounding_box if BR is None else BoundingBox3D.union(BR, p.bounding_box) 
                
                if NR == 0 or NL == 0:
                    continue 

                cost = BL.area * NL + BR.area * NR 
                if cost < best_cost: 
                    best_cost = cost
                    best = (axis, thresh)

        return best

    
    def rearrange(self, start, end, axis, threshold):
        i = start
        j = end - 1

        while i <= j:
            if self.primitives[i].bounding_box.center[axis] <= threshold:
                i += 1
            else:
                self.primitives[i], self.primitives[j] = self.primitives[j], self.primitives[i]
                j -= 1

        return i
            
    def build_bfs(self):
        # create root
        root = BVHNode(prim_left=0, prim_right=len(self.primitives), depth=0)
        self.nodes.append(root)

        q: Deque[WorkItem] = deque()
        q.append(WorkItem(0, 0, len(self.primitives), 0)) # enqueue the first node (id = 0, start = 0, end = length, depth = 0)

        while q and len(self.nodes) < self.max_nodes: 
            item = q.popleft()
            node = self.nodes[item.node_idx]
            start, end, depth = item.start, item.end, item.depth
            n = end - start 

            if self.on_progress is not None: 
                self.on_progress(len(self.nodes), self.max_nodes)
            
            if n <= 0:
                continue 

            bbox = self.primitives[start].bounding_box
            for p in self.primitives[start+1 : end]:
                bbox = BoundingBox3D.union(bbox, p.bounding_box)
            node.bound = bbox

            # leaf condition
            if (n <= self.min_prim_per_node or depth >= self.max_depth or len(self.nodes) >= self.max_nodes):
                node.left = -1
                node.right = -1
                continue 

            split = self.SAH_split(start, end)
            if split is None:
                continue
            axis, threshold = split

            mid = self.rearrange(start, end, axis, threshold)
            if mid == start or mid == end:
                continue 

            left_idx = len(self.nodes)
            self.nodes.append(BVHNode(prim_left=start, prim_right=mid, depth=depth + 1))
            right_idx = len(self.nodes)
            self.nodes.append(BVHNode(prim_left=mid, prim_right=end, depth=depth + 1))

            node.left = left_idx
            node.right = right_idx

            q.append(WorkItem(left_idx, start, mid, depth + 1))
            q.append(WorkItem(right_idx, mid, end, depth + 1))
            
        # TODO: Student implementation ends here.


def create_bvh_node_buf(module: spy.Module, bvh_nodes: List[BVHNode]) -> spy.NDBuffer:
    device = module.device
    node_buf = spy.NDBuffer(
        device=device, dtype=module.BVHNode.as_struct(), shape=(max(len(bvh_nodes), 1),)
    )
    cursor = node_buf.cursor()
    for idx, node in enumerate(bvh_nodes):
        cursor[idx].write(node.get_this())
    cursor.apply()
    return node_buf
