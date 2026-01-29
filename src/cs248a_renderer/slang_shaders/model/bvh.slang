implementing "../model.slang";
import math;

public struct BVHNode
{
    public BoundingBox bound;
    public int left;
    public int right;
    public int primLeft;
    public int primRight;
    public int depth;

    public __init(BoundingBox bound, int left = -1, int right = -1, int primLeft = -1, int primRight = -1, int depth = -1)
    {
        this.bound = bound;
        this.left = left;
        this.right = right;
        this.primLeft = primLeft;
        this.primRight = primRight;
        this.depth = depth;
    }

    public float2 hit(Ray ray)
    {
        return this.bound.hit(ray);
    }
}

public struct BVH<T>
    where T : IRayCastable
{
    public struct HitResult
    {
        public T primitive;
        public RayHitResult rayHitResult;
    }

    // Slang buffer of BVHNode copied from Python.
    public RWStructuredBuffer<BVHNode> nodes;
    // Maximum number of nodes in the BVH, in this case, it'll be the size of the nodes buffer.
    public uint maxNodes;
    // Slang buffer of primitives represented by the BVH.
    public RWStructuredBuffer<T> primitives;
    // Number of primitives in the primitives buffer.
    public uint numPrimitives;

    public __init(RWStructuredBuffer<BVHNode> nodes, RWStructuredBuffer<T> primitives, uint numPrimitives, uint maxNodes)
    {
        this.nodes = nodes;
        this.primitives = primitives;
        this.numPrimitives = numPrimitives;
        this.maxNodes = maxNodes;
    }

    struct HitStack<int N>
    {
        uint[N] nodeIndices;
        uint top;

        __init()
        {
            this.top = 0;
        }

        [mutating]
        void push(uint nodeIndex)
        {
            this.nodeIndices[this.top] = nodeIndex;
            this.top += 1;
        }

        [mutating]
        uint pop()
        {
            this.top -= 1;
            return this.nodeIndices[this.top];
        }

        bool isEmpty()
        {
            return this.top == 0;
        }

        uint size()
        {
            return this.top;
        }
    }

    // BVH traversal to find the closest hit.
    public Optional<HitResult> hit(Ray ray){
        // Initialize traversal.
        Optional<HitResult> hitResult = Optional<HitResult>();
        HitStack<64> stack;

        // TODO: Student implementation starts here.

        float bestT = ray.tRange.y; 

        stack.push(0);

        while(!stack.isEmpty()){
            uint node_idx = stack.pop();
            BVHNode node = nodes[node_idx];
            
            float2 tbox = node.hit(ray);
            float tEnter = tbox.x;
            float tExit = tbox.y;

            if (tEnter > tExit) continue;
            if (tExit < ray.tRange.x) continue;
            if (tEnter > bestT) continue;

            bool isLeaf = (node.left == -1) && (node.right == -1);

            if (isLeaf){
                int start = node.primLeft;
                int end = node.primRight;
                if (start < 0 || end < 0 ) continue;

                for (int i = start; i < end; ++i){
                    Optional<RayHitResult> pr = primitives[i].hit(ray);
                    if (!pr.hasValue) continue;

                    RayHitResult h = pr.value;
                    if (h.t>=bestT) continue;
                    if (h.t<ray.tRange.x) continue;
                    bestT = h.t;
                    HitResult HIT;
                    HIT.primitive = primitives[i];
                    HIT.rayHitResult = h;
                    hitResult = Optional<HitResult>(HIT);
                }
            }else{
                int l = node.left;
                int r = node.right;
                
                float2 to_l = nodes[l].hit(ray);
                float2 to_r = nodes[r].hit(ray);
                bool hitL = (to_l.x <= to_l.y) && (to_l.y >= ray.tRange.x) && (to_l.x <= bestT);
                bool hitR = (to_r.x <= to_r.y) && (to_r.y >= ray.tRange.x) && (to_r.x <= bestT);

                if (!hitL && !hitR) continue;

                if (hitL && hitR){
                    if (to_l.x < to_r.x){
                        stack.push((uint)r);
                        stack.push((uint)l);
                    }else{
                        stack.push((uint)l);
                        stack.push((uint)r);
                    }
                }
                else if (hitL){
                    stack.push((uint)l);
                }
                else {
                    stack.push((uint)r);
                }
            }
        }

        // TODO: Student implementation ends here.

        return hitResult;
    }
}
