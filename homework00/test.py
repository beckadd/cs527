class LinkedList:
    next = None
    value = None

    def __init__(self, value = None, next = None):
       self.next, self.value = next, value 

    def __str__(self):
        strg = ""
        node = self
        while node != None:
            if node.next == None:
                strg += str(node.value)
            else:
                strg += str(node.value) + "->"
            node = node.next
        return strg

def link_nodes(nodes):
    next_node = None;
    curr = None
    for i in range(len(nodes)-1, -1, -1):
        node = nodes[i]
        curr = LinkedList(value = node, next = next_node)
        next_node = curr
    return curr

def reverse(head):
    if head == None or head.next == None:
        return head
    new_head = reverse(head.next)
    head.next.next = head
    head.next = None
    return new_head


nodes = link_nodes(range(0,100))
