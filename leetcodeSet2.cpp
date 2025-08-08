#include <iostream>
#include <vector>
#include <stack>
#include <queue>
#include <algorithm>
#include <string>
using namespace std;

// 1. Merge Two Sorted Arrays
void merge(vector<int>& nums1, int m, vector<int>& nums2, int n) {
    int i = m - 1, j = n - 1, k = m + n - 1;
    while (j >= 0) {
        nums1[k--] = (i >= 0 && nums1[i] > nums2[j]) ? nums1[i--] : nums2[j--];
    }
}

// 2. Valid Parentheses
bool isValid(string s) {
    stack<char> st;
    for (char c : s) {
        if (c == '(' || c == '{' || c == '[') st.push(c);
        else {
            if (st.empty()) return false;
            char t = st.top(); st.pop();
            if ((c == ')' && t != '(') || (c == ']' && t != '[') || (c == '}' && t != '{'))
                return false;
        }
    }
    return st.empty();
}

// 3. Remove Element
int removeElement(vector<int>& nums, int val) {
    int idx = 0;
    for (int n : nums) {
        if (n != val) nums[idx++] = n;
    }
    return idx;
}

// 4. Linked List Cycle
struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(nullptr) {}
};

bool hasCycle(ListNode *head) {
    ListNode *slow = head, *fast = head;
    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
        if (slow == fast) return true;
    }
    return false;
}

// 5. Maximum Subarray
int maxSubArray(vector<int>& nums) {
    int maxSum = nums[0], cur = nums[0];
    for (int i = 1; i < nums.size(); ++i) {
        cur = max(nums[i], cur + nums[i]);
        maxSum = max(maxSum, cur);
    }
    return maxSum;
}

// 6. Binary Search
int binarySearch(vector<int>& nums, int target) {
    int left = 0, right = nums.size() - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] == target) return mid;
        else if (nums[mid] < target) left = mid + 1;
        else right = mid - 1;
    }
    return -1;
}

// 7. Power of Two
bool isPowerOfTwo(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}

// 8. Invert Binary Tree
struct TreeNode {
    int val;
    TreeNode *left, *right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

TreeNode* invertTree(TreeNode* root) {
    if (!root) return nullptr;
    swap(root->left, root->right);
    invertTree(root->left);
    invertTree(root->right);
    return root;
}

// 9. Implement a Queue Using Stacks
class MyQueue {
    stack<int> in, out;
    void move() {
        if (out.empty()) while (!in.empty()) out.push(in.top()), in.pop();
    }
public:
    void push(int x) { in.push(x); }
    int pop() { move(); int x = out.top(); out.pop(); return x; }
    int peek() { move(); return out.top(); }
    bool empty() { return in.empty() && out.empty(); }
};

// 10. Majority Element
int majorityElement(vector<int>& nums) {
    int count = 0, candidate = 0;
    for (int n : nums) {
        if (count == 0) candidate = n;
        count += (n == candidate) ? 1 : -1;
    }
    return candidate;
}

// Helper function to print vector
template<typename T>
void printVector(const vector<T>& v, int len = -1) {
    if (len == -1) len = v.size();
    cout << "[";
    for (int i = 0; i < len; ++i) {
        cout << v[i];
        if (i != len-1) cout << ", ";
    }
    cout << "]";
}

int main() {
    // 1. Merge Two Sorted Arrays
    cout << "1. Merge Two Sorted Arrays: ";
    vector<int> nums1 = {1,2,3,0,0,0}, nums2 = {2,5,6};
    merge(nums1, 3, nums2, 3);
    printVector(nums1); cout << endl;

    // 2. Valid Parentheses
    cout << "2. Valid Parentheses: ";
    cout << (isValid("({[]})") ? "true" : "false") << endl;

    // 3. Remove Element
    cout << "3. Remove Element: ";
    vector<int> nums3 = {3,2,2,3};
    int len3 = removeElement(nums3, 3);
    printVector(nums3, len3); cout << " (new length: " << len3 << ")" << endl;

    // 4. Linked List Cycle
    cout << "4. Linked List Cycle: ";
    ListNode* a = new ListNode(1);
    a->next = new ListNode(2);
    a->next->next = a; // cycle
    cout << (hasCycle(a) ? "true" : "false") << endl;
    a->next->next = nullptr; // break cycle
    delete a->next; delete a;

    // 5. Maximum Subarray
    cout << "5. Maximum Subarray: ";
    vector<int> nums5 = {-2,1,-3,4,-1,2,1,-5,4};
    cout << maxSubArray(nums5) << endl;

    // 6. Binary Search
    cout << "6. Binary Search: ";
    vector<int> nums6 = {-1,0,3,5,9,12};
    cout << binarySearch(nums6, 9) << endl;

    // 7. Power of Two
    cout << "7. Power of Two: ";
    cout << (isPowerOfTwo(16) ? "true" : "false") << endl;

    // 8. Invert Binary Tree
    cout << "8. Invert Binary Tree: ";
    TreeNode* root = new TreeNode(4);
    root->left = new TreeNode(2);
    root->right = new TreeNode(7);
    root->left->left = new TreeNode(1);
    root->left->right = new TreeNode(3);
    root->right->left = new TreeNode(6);
    root->right->right = new TreeNode(9);
    invertTree(root);
    // Print root after inversion (left and right children swapped)
    cout << "Root Left: " << root->left->val << ", Root Right: " << root->right->val << endl;
    // Clean up
    delete root->right->right; delete root->right->left; delete root->left->right; delete root->left->left; delete root->left; delete root->right; delete root;

    // 9. Implement a Queue Using Stacks
    cout << "9. Implement a Queue Using Stacks: ";
    MyQueue q;
    q.push(1); q.push(2);
    cout << "Front: " << q.peek() << ", Popped: " << q.pop() << ", Empty: " << (q.empty() ? "true" : "false") << endl;

    // 10. Majority Element
    cout << "10. Majority Element: ";
    vector<int> nums10 = {2,2,1,1,1,2,2};
    cout << majorityElement(nums10) << endl;

    return 0;
}