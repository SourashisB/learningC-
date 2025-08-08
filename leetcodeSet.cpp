#include <iostream>
#include <vector>
#include <unordered_map>
#include <list>
#include <fstream>
#include <string>
#include <mutex>
#include <stdexcept>
#include <queue>
using namespace std;


// 1. Reverse a String In-place
void reverseString(string &s) {
    int left = 0, right = s.length() - 1;
    while (left < right) {
        swap(s[left], s[right]);
        ++left;
        --right;
    }
}

// 2. Remove Duplicates from Sorted Array
int removeDuplicates(vector<int>& nums) {
    if (nums.empty()) return 0;
    int idx = 1;
    for (int i = 1; i < nums.size(); ++i) {
        if (nums[i] != nums[i-1]) {
            nums[idx++] = nums[i];
        }
    }
    return idx;
}

// 3. Two Sum
vector<int> twoSum(vector<int>& nums, int target) {
    unordered_map<int, int> seen;
    for (int i = 0; i < nums.size(); ++i) {
        int complement = target - nums[i];
        if (seen.count(complement)) {
            return {seen[complement], i};
        }
        seen[nums[i]] = i;
    }
    return {};
}

// 4. Implement a Simple LRU Cache
class LRUCache {
    int cap;
    list<pair<int, int>> items; // key, value
    unordered_map<int, list<pair<int, int>>::iterator> cache;
public:
    LRUCache(int capacity) : cap(capacity) {}

    int get(int key) {
        if (cache.find(key) == cache.end()) return -1;
        auto it = cache[key];
        items.splice(items.begin(), items, it);
        return it->second;
    }

    void put(int key, int value) {
        if (cache.find(key) != cache.end()) {
            auto it = cache[key];
            it->second = value;
            items.splice(items.begin(), items, it);
            return;
        }
        if (items.size() == cap) {
            int oldKey = items.back().first;
            cache.erase(oldKey);
            items.pop_back();
        }
        items.emplace_front(key, value);
        cache[key] = items.begin();
    }
};

// 5. Find the Intersection Node of Two Linked Lists
struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(nullptr) {}
};

ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
    if (!headA || !headB) return nullptr;
    ListNode *a = headA, *b = headB;
    while (a != b) {
        a = a ? a->next : headB;
        b = b ? b->next : headA;
    }
    return a;
}

// 6. Implement a Singleton Class
class Singleton {
    Singleton() {}
    Singleton(const Singleton&) = delete;
    Singleton& operator=(const Singleton&) = delete;
public:
    static Singleton& getInstance() {
        static Singleton instance;
        return instance;
    }
    void hello() { cout << "Hello from Singleton!" << endl; }
};

// 7. Count Occurrences of a Character in a Text File
int countCharInFile(const string& filename, char target) {
    ifstream file(filename);
    if (!file.is_open()) return -1;
    int count = 0;
    char c;
    while (file.get(c)) {
        if (c == target) ++count;
    }
    return count;
}

// 8. Implement a Stack using Queues
class MyStack {
    queue<int> q;
public:
    void push(int x) {
        q.push(x);
        for (int i = 0; i < q.size() - 1; ++i) {
            q.push(q.front());
            q.pop();
        }
    }
    int pop() {
        int top = q.front();
        q.pop();
        return top;
    }
    int top() {
        return q.front();
    }
    bool empty() {
        return q.empty();
    }
};

int main() {
    cout << "1. Reverse String Example:" << endl;
    string s = "hello";
    reverseString(s);
    cout << "Reversed: " << s << endl << endl;

    cout << "2. Remove Duplicates from Sorted Array Example:" << endl;
    vector<int> nums = {1, 1, 2, 2, 3};
    int newLen = removeDuplicates(nums);
    cout << "New Length: " << newLen << "; Array: ";
    for (int i = 0; i < newLen; ++i) cout << nums[i] << " ";
    cout << endl << endl;

    cout << "3. Two Sum Example:" << endl;
    vector<int> nums2 = {2, 7, 11, 15};
    vector<int> res = twoSum(nums2, 9);
    cout << "Indices: ";
    for (int i : res) cout << i << " ";
    cout << endl << endl;

    cout << "4. LRU Cache Example:" << endl;
    LRUCache cache(2);
    cache.put(1, 1);
    cache.put(2, 2);
    cout << "Get 1: " << cache.get(1) << endl; // returns 1
    cache.put(3, 3);                            // evicts key 2
    cout << "Get 2: " << cache.get(2) << endl; // returns -1 (not found)
    cache.put(4, 4);                            // evicts key 1
    cout << "Get 1: " << cache.get(1) << endl; // returns -1 (not found)
    cout << "Get 3: " << cache.get(3) << endl; // returns 3
    cout << "Get 4: " << cache.get(4) << endl << endl; // returns 4

    cout << "5. Intersection Node of Two Linked Lists Example:" << endl;
    // Create two lists that intersect
    ListNode* common = new ListNode(8);
    common->next = new ListNode(10);
    ListNode* headA = new ListNode(3);
    headA->next = new ListNode(7);
    headA->next->next = common;
    ListNode* headB = new ListNode(99);
    headB->next = common;
    ListNode* intersection = getIntersectionNode(headA, headB);
    cout << "Intersection Node Value: " << (intersection ? to_string(intersection->val) : "null") << endl << endl;
    delete headA->next; delete headA; delete headB; delete common->next; delete common;

    cout << "6. Singleton Example:" << endl;
    Singleton::getInstance().hello();
    cout << endl;

    cout << "7. Count Character in File Example:" << endl;
    // For demonstration, we write to a file first.
    ofstream testFile("test.txt");
    testFile << "abracadabra";
    testFile.close();
    int count = countCharInFile("test.txt", 'a');
    cout << "Count of 'a' in test.txt: " << count << endl << endl;

    cout << "8. Stack using Queues Example:" << endl;
    MyStack stack;
    stack.push(1);
    stack.push(2);
    cout << "Top: " << stack.top() << endl; // returns 2
    cout << "Pop: " << stack.pop() << endl; // returns 2
    cout << "Empty: " << (stack.empty() ? "true" : "false") << endl;

    return 0;
}