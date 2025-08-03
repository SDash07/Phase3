import random
import time
import sys

# increase recursion limit so AVL delete (which is recursive) can handle larger trees safely
sys.setrecursionlimit(20000)

# ---------------------------
# Phase 2: Unbalanced BST (with iterative insert/search)
# ---------------------------
class BSTNode:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None

class BST:
    def __init__(self):
        self.root = None

    def insert(self, key):
        if self.root is None:
            self.root = BSTNode(key)
            return
        current = self.root
        while True:
            if key < current.key:
                if current.left is None:
                    current.left = BSTNode(key)
                    return
                current = current.left
            elif key > current.key:
                if current.right is None:
                    current.right = BSTNode(key)
                    return
                current = current.right
            else:
                # duplicate: ignore
                return

    def search(self, key):
        current = self.root
        while current:
            if key == current.key:
                return True
            elif key < current.key:
                current = current.left
            else:
                current = current.right
        return False

    def delete(self, key):
        self.root, deleted = self._delete_recursive(self.root, key)
        return deleted

    def _delete_recursive(self, node, key):
        if node is None:
            return node, False
        if key < node.key:
            node.left, deleted = self._delete_recursive(node.left, key)
            return node, deleted
        elif key > node.key:
            node.right, deleted = self._delete_recursive(node.right, key)
            return node, deleted
        # node to delete found
        if node.left is None and node.right is None:
            return None, True
        elif node.left is None:
            return node.right, True
        elif node.right is None:
            return node.left, True
        else:
            successor = node.right
            while successor.left:
                successor = successor.left
            node.key = successor.key
            node.right, _ = self._delete_recursive(node.right, successor.key)
            return node, True

    def inorder_traversal(self):
        res = []
        stack = []
        current = self.root
        while stack or current:
            while current:
                stack.append(current)
                current = current.left
            current = stack.pop()
            res.append(current.key)
            current = current.right
        return res

# ---------------------------
# Phase 3: Optimized AVL Tree
# ---------------------------
class AVLNode:
    __slots__ = ('key', 'left', 'right', 'height')
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None
        self.height = 1

def height(node):
    return node.height if node else 0

def update_height(node):
    node.height = 1 + max(height(node.left), height(node.right))

def balance_factor(node):
    return height(node.left) - height(node.right) if node else 0

def rotate_right(y):
    x = y.left
    T2 = x.right
    x.right = y
    y.left = T2
    update_height(y)
    update_height(x)
    return x

def rotate_left(x):
    y = x.right
    T2 = y.left
    y.left = x
    x.right = T2
    update_height(x)
    update_height(y)
    return y

class AVLTree:
    def __init__(self):
        self.root = None

    def insert(self, key):
        self.root = self._insert(self.root, key)

    def _insert(self, node, key):
        if not node:
            return AVLNode(key)
        if key < node.key:
            node.left = self._insert(node.left, key)
        elif key > node.key:
            node.right = self._insert(node.right, key)
        else:
            return node  # no duplicates
        update_height(node)
        bf = balance_factor(node)
        if bf > 1:
            if key < node.left.key:
                return rotate_right(node)
            else:
                node.left = rotate_left(node.left)
                return rotate_right(node)
        if bf < -1:
            if key > node.right.key:
                return rotate_left(node)
            else:
                node.right = rotate_right(node.right)
                return rotate_left(node)
        return node

    def search(self, key):
        return self._search(self.root, key)

    def _search(self, node, key):
        if not node:
            return False
        if key == node.key:
            return True
        elif key < node.key:
            return self._search(node.left, key)
        else:
            return self._search(node.right, key)

    def inorder(self):
        res = []
        self._inorder(self.root, res)
        return res

    def _inorder(self, node, res):
        if node:
            self._inorder(node.left, res)
            res.append(node.key)
            self._inorder(node.right, res)

    def delete(self, key):
        self.root = self._delete(self.root, key)

    def _delete(self, node, key):
        if not node:
            return node
        if key < node.key:
            node.left = self._delete(node.left, key)
        elif key > node.key:
            node.right = self._delete(node.right, key)
        else:
            if not node.left:
                return node.right
            elif not node.right:
                return node.left
            temp = node.right
            while temp.left:
                temp = temp.left
            node.key = temp.key
            node.right = self._delete(node.right, temp.key)
        update_height(node)
        bf = balance_factor(node)
        if bf > 1:
            if balance_factor(node.left) >= 0:
                return rotate_right(node)
            else:
                node.left = rotate_left(node.left)
                return rotate_right(node)
        if bf < -1:
            if balance_factor(node.right) <= 0:
                return rotate_left(node)
            else:
                node.right = rotate_right(node.right)
                return rotate_left(node)
        return node

# ---------------------------
# Benchmarking / Scaling / Validation
# ---------------------------
def count_nodes(node):
    if not node:
        return 0
    left = count_nodes(node.left) if hasattr(node, 'left') else 0
    right = count_nodes(node.right) if hasattr(node, 'right') else 0
    return 1 + left + right

def measure(tree_cls, data, sample_size=100):
    tree = tree_cls()
    t0 = time.perf_counter()
    for x in data:
        tree.insert(x)
    t1 = time.perf_counter()
    insert_time = t1 - t0

    existing = random.sample(data, min(sample_size, len(data)))
    missing = [max(data) + i + 1 for i in range(len(existing))]
    search_keys = existing + missing
    random.shuffle(search_keys)
    t0 = time.perf_counter()
    for k in search_keys:
        tree.search(k)
    t1 = time.perf_counter()
    search_time = t1 - t0

    delete_keys = random.sample(data, min(sample_size, len(data)))
    t0 = time.perf_counter()
    for k in delete_keys:
        tree.delete(k)
    t1 = time.perf_counter()
    delete_time = t1 - t0

    size = count_nodes(tree.root)
    return insert_time, search_time, delete_time, size

def scaling_test(sizes):
    results = []
    for n in sizes:
        data_random = random.sample(range(n * 3), n)
        bst_metrics = measure(BST, data_random)
        avl_metrics = measure(AVLTree, data_random)
        results.append((n, bst_metrics, avl_metrics))
    return results

# ---------------------------
# Execution and Comparison
# ---------------------------
if __name__ == "__main__":
    random.seed(42)
    sizes = [1000, 5000, 10000]
    results = scaling_test(sizes)
    for n, bst_m, avl_m in results:
        print(f"Dataset size: {n}")
        print(f" Unbalanced BST: insert={bst_m[0]:.4f}s, search={bst_m[1]:.4f}s, delete={bst_m[2]:.4f}s, nodes={bst_m[3]}")
        print(f" AVL Tree:       insert={avl_m[0]:.4f}s, search={avl_m[1]:.4f}s, delete={avl_m[2]:.4f}s, nodes={avl_m[3]}")
        print("-" * 60)

    # Worst-case sequential insertion (sorted) for BST vs AVL
    print("\nWorst-case sequential insertion (sorted):")
    seq = list(range(5000))
    bst_seq = BST()
    avl_seq = AVLTree()

    t0 = time.perf_counter()
    for x in seq:
        bst_seq.insert(x)
    t1 = time.perf_counter()
    bst_seq_time = t1 - t0

    t0 = time.perf_counter()
    for x in seq:
        avl_seq.insert(x)
    t1 = time.perf_counter()
    avl_seq_time = t1 - t0

    print(f" Sequential insert 5000 sorted keys: BST={bst_seq_time:.4f}s, AVL={avl_seq_time:.4f}s")
