from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import NoSuchElementException, ElementClickInterceptedException
import random
import sys
import time
import string
from selenium.webdriver.common.keys import Keys

class Test_module:
    def __init__(self):
        self.google = "https://www.google.com/"
        self.base_url = "http://localhost:8000"
        self.name=['Ken adams','steve rojers','barry dave','ross geller','joey tribbiani','chandler bing','chris moris','vinod joshi']
        self.designation=['fever','cold','sleep','stress','pain']
        self.contact=[9089988767,8998877898,9890988767,8765456765,9865432345]
        self.address=['kopargaon','shirdi','ahmednagar','pune','mumbai']
        self.f_name=['soham','prasad','vishal','nikhil','rushi','shubham']
        self.l_name=['kale','chaudari','gore','dhavle','shinde','wagh','gavhane']
        self.medi=['adalimumab', 'apixaban', 'lenalidomide', 'nivolumab', 'pembrolizumab', 'etanercept', 'trastuzumab', 'bevacizumab','rituximab', 'rivaroxaban']
        self.driver = webdriver.Chrome(ChromeDriverManager().install())
        self.COMMAND_OR_CONTROL = Keys.COMMAND if sys.platform == 'darwin' else Keys.CONTROL
    def open_url(self):
        self.driver.get(self.base_url)

    def random_char(self,y):
       return ''.join(random.choice(string.ascii_letters) for x in range(y))

    def Add_medi(self):
        time.sleep(1)
        p = self.driver.find_element_by_link_text("Medicine Management").click()
        time.sleep(2)
        p = self.driver.find_element_by_link_text("Add Medicine Details").click()
        time.sleep(2)
        self.driver.find_element_by_name("mid").send_keys(random.randint(1,50))
        time.sleep(1)
        self.driver.find_element_by_name("mname").send_keys(random.choice(self.medi))
        time.sleep(1)
        self.driver.find_element_by_name("dname").send_keys(random.choice(self.name))
        time.sleep(1)
        
        self.driver.find_element_by_name("price").send_keys(random.randint(10,100))
        time.sleep(1)
        self.driver.find_element_by_name("stock").send_keys(random.randint(10,100))
        time.sleep(1)
        self.driver.find_element_by_name("desc").send_keys(random.choice(self.designation))
        time.sleep(1)
        p = self.driver.find_element_by_name('add-record').click()
        time.sleep(2)
    
    
    def View_medi(self):
        time.sleep(1)
        p = self.driver.find_element_by_link_text("Medicine Management").click()
        time.sleep(2)
        p = self.driver.find_element_by_link_text("View Medicine Details").click()
        time.sleep(2)
    def Add_emp(self):
        time.sleep(1)
        p = self.driver.find_element_by_link_text("Employee Management").click()
        time.sleep(2)
        p = self.driver.find_element_by_link_text("Add Employee Details").click()
        time.sleep(2)
        self.driver.find_element_by_name("eid").send_keys(random.randint(1,50))
        time.sleep(1)
        self.driver.find_element_by_name("fname").send_keys(random.choice(self.f_name))
        time.sleep(1)
        self.driver.find_element_by_name("lname").send_keys(random.choice(self.l_name))
        time.sleep(1)
        self.driver.find_element_by_name("address").send_keys(random.choice(self.address))
        time.sleep(1)
        self.driver.find_element_by_name("sal").send_keys(random.randint(10000,70000))
        time.sleep(1)
        self.driver.find_element_by_name("pno").send_keys(random.choice(self.contact))
        time.sleep(1)
        self.driver.find_element_by_name("email").send_keys(bot.random_char(7)+"@gmail.com")
        time.sleep(1)
        p = self.driver.find_element_by_name('add-record').click()
        time.sleep(2)
    def View_emp(self):
        time.sleep(1)
        p = self.driver.find_element_by_link_text("Employee Management").click()
        time.sleep(2)
        p = self.driver.find_element_by_link_text("View Employee Details").click()
        time.sleep(2)

    def Add_dealer(self):
        time.sleep(1)
        p = self.driver.find_element_by_link_text("Dealer Management").click()
        time.sleep(2)
        p = self.driver.find_element_by_link_text("Add Dealer Details").click()
        time.sleep(2)
        self.driver.find_element_by_name("dname").send_keys(random.choice(self.name))
        time.sleep(1)
        self.driver.find_element_by_name("address").send_keys(random.choice(self.address))
        time.sleep(1)
        self.driver.find_element_by_name("pno").send_keys(random.choice(self.contact))
        time.sleep(1)
        
        self.driver.find_element_by_name("email").send_keys(bot.random_char(7)+"@gmail.com")
        time.sleep(1)
        p = self.driver.find_element_by_name('cancle').click()
        time.sleep(2)
    
    
    
if __name__ == '__main__':
    bot = Test_module()
    bot.open_url()
    bot.Add_medi()
    bot.View_medi()
    bot.Add_emp()
    bot.View_emp()
    bot.Add_dealer()



