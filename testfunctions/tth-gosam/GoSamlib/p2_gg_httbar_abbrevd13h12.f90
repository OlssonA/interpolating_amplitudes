module     p2_gg_httbar_abbrevd13h12
   use p2_gg_httbar_config, only: ki
   use p2_gg_httbar_kinematics, only: epstensor
   use p2_gg_httbar_globalsh12
   implicit none
   private
   complex(ki), dimension(60), public :: abb13
   complex(ki), public :: R2d13
   public :: init_abbrev
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
contains
   subroutine     init_abbrev()
      use p2_gg_httbar_config, only: deltaOS, &
     &    logfile, debug_nlo_diagrams
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_model
      use p2_gg_httbar_color, only: TR
      use p2_gg_httbar_globalsl1, only: epspow
      implicit none
      abb13(1)=1.0_ki/(mH**2+mT**2-es34-es45+es12)
      abb13(2)=es12**(-1)
      abb13(3)=spak2l4**(-1)
      abb13(4)=spak2l3**(-1)
      abb13(5)=spbl3k2**(-1)
      abb13(6)=spak2l5**(-1)
      abb13(7)=sqrt(mT**2)
      abb13(8)=spbl4k2**(-1)
      abb13(9)=c1-c2
      abb13(9)=abb13(9)*spae1e2*NC*gs**4*i_*TR*spbe2e1*e*gHT*abb13(1)
      abb13(10)=abb13(2)*abb13(9)
      abb13(11)=-mT*abb13(10)
      abb13(12)=abb13(11)*abb13(7)
      abb13(13)=abb13(3)*spak1k2
      abb13(14)=abb13(12)*abb13(13)
      abb13(15)=mT**2
      abb13(16)=-abb13(15)*abb13(10)
      abb13(17)=abb13(16)*abb13(13)
      abb13(18)=abb13(14)+abb13(17)
      abb13(19)=abb13(18)*spbl5k1
      abb13(20)=spak2l3*abb13(6)
      abb13(21)=abb13(17)*abb13(20)
      abb13(22)=abb13(21)*spbl3k1
      abb13(22)=abb13(22)+abb13(19)
      abb13(23)=abb13(16)*spak1k2
      abb13(24)=abb13(7)*spak1k2
      abb13(25)=abb13(11)*abb13(24)
      abb13(26)=abb13(23)+abb13(25)
      abb13(27)=spbl4k1*abb13(6)
      abb13(28)=abb13(26)*abb13(27)
      abb13(29)=mH**2*abb13(5)*abb13(4)
      abb13(30)=abb13(29)*spbl5k2
      abb13(31)=abb13(30)*spak1k2
      abb13(32)=abb13(10)*spbl4k1
      abb13(33)=abb13(31)*abb13(32)
      abb13(34)=abb13(10)*spbl5l3
      abb13(35)=abb13(34)*spbl4k1
      abb13(36)=abb13(35)*spak1l3
      abb13(36)=abb13(33)+abb13(36)-abb13(28)
      abb13(37)=abb13(34)*spbl4k2
      abb13(38)=abb13(37)*spak2l3
      abb13(39)=-abb13(38)+abb13(36)
      abb13(40)=abb13(39)-abb13(22)
      abb13(41)=-1.0_ki/4.0_ki*abb13(40)
      abb13(26)=abb13(26)*abb13(6)
      abb13(42)=spbl4k1**2
      abb13(43)=-abb13(42)*abb13(26)
      abb13(19)=-spbl4k1*abb13(19)
      abb13(44)=3.0_ki*spbl4k1
      abb13(45)=-abb13(38)*abb13(44)
      abb13(19)=abb13(19)+abb13(43)+abb13(45)
      abb13(19)=spak1l4*abb13(19)
      abb13(43)=abb13(23)-abb13(25)
      abb13(45)=abb13(20)*spbl3k2
      abb13(43)=abb13(43)*abb13(45)
      abb13(46)=abb13(10)*abb13(7)**2
      abb13(47)=spak1k2*abb13(46)
      abb13(23)=abb13(23)+abb13(47)
      abb13(23)=spbl5k2*abb13(23)
      abb13(23)=abb13(43)+abb13(23)
      abb13(23)=abb13(44)*abb13(23)
      abb13(43)=mT**4
      abb13(44)=-abb13(43)*abb13(9)
      abb13(47)=-abb13(44)*abb13(13)
      abb13(48)=mT**3
      abb13(49)=-abb13(48)*abb13(9)
      abb13(50)=abb13(13)*abb13(7)
      abb13(51)=-abb13(49)*abb13(50)
      abb13(47)=abb13(47)+abb13(51)
      abb13(47)=abb13(8)*abb13(47)*abb13(27)
      abb13(51)=abb13(3)**2
      abb13(52)=abb13(44)*abb13(51)
      abb13(53)=abb13(52)*spak1k2
      abb13(54)=abb13(49)*abb13(51)
      abb13(55)=-abb13(24)*abb13(54)
      abb13(53)=-abb13(53)+abb13(55)
      abb13(53)=abb13(8)*abb13(53)
      abb13(55)=abb13(10)*abb13(24)
      abb13(56)=-spak1k2*abb13(11)
      abb13(55)=abb13(56)+abb13(55)
      abb13(56)=3.0_ki*spbl4k2
      abb13(55)=abb13(7)*abb13(55)*abb13(56)
      abb13(53)=abb13(55)+abb13(53)
      abb13(53)=spbl5k1*abb13(53)
      abb13(15)=-abb13(15)*abb13(9)
      abb13(55)=abb13(8)*abb13(15)*spbl5k2
      abb13(57)=-spbl4k1*abb13(13)*abb13(55)
      abb13(42)=abb13(42)*spak1l4
      abb13(58)=abb13(10)*spbl5k2
      abb13(59)=spak1k2*abb13(58)*abb13(42)
      abb13(57)=abb13(57)+abb13(59)
      abb13(57)=abb13(57)*abb13(29)
      abb13(59)=abb13(8)*spbl5l3
      abb13(15)=-abb13(3)*abb13(59)*abb13(15)*spbl4k1
      abb13(42)=abb13(34)*abb13(42)
      abb13(15)=abb13(15)+abb13(42)
      abb13(15)=spak1l3*abb13(15)
      abb13(28)=-3.0_ki*abb13(28)-abb13(38)
      abb13(28)=spbl4k2*abb13(28)
      abb13(33)=abb13(33)*abb13(56)
      abb13(42)=3.0_ki*abb13(37)
      abb13(60)=abb13(42)*spak1l3*spbl4k1
      abb13(28)=abb13(60)+abb13(28)+abb13(33)
      abb13(28)=spak2l4*abb13(28)
      abb13(25)=-abb13(25)*abb13(56)
      abb13(33)=abb13(52)*abb13(8)
      abb13(56)=-spak1k2*abb13(33)
      abb13(25)=abb13(25)+abb13(56)
      abb13(25)=abb13(20)*abb13(25)
      abb13(56)=-spak1l4*spbl4k1*abb13(21)
      abb13(25)=abb13(56)+abb13(25)
      abb13(25)=spbl3k1*abb13(25)
      abb13(56)=spbl5l3*spak2l3
      abb13(9)=-abb13(7)*abb13(3)*abb13(56)*mT*abb13(9)
      abb13(9)=abb13(25)+abb13(28)+abb13(15)+abb13(57)+abb13(19)+abb13(53)+3.0_&
      &ki*abb13(9)+abb13(47)+abb13(23)
      abb13(9)=1.0_ki/4.0_ki*abb13(9)
      abb13(15)=1.0_ki/2.0_ki*abb13(39)
      abb13(19)=-abb13(43)*abb13(10)
      abb13(13)=-abb13(19)*abb13(13)
      abb13(23)=abb13(10)*abb13(48)
      abb13(25)=abb13(23)*abb13(50)
      abb13(13)=abb13(13)+abb13(25)
      abb13(13)=abb13(13)*abb13(27)
      abb13(23)=abb13(23)*abb13(51)*abb13(24)
      abb13(19)=abb13(19)*abb13(51)*spak1k2
      abb13(23)=-abb13(19)+abb13(23)
      abb13(23)=spbl5k1*abb13(23)
      abb13(24)=spbl4k1*abb13(17)*abb13(30)
      abb13(19)=spbl3k1*abb13(20)*abb13(19)
      abb13(13)=-abb13(13)-abb13(23)+abb13(24)+abb13(19)
      abb13(13)=-1.0_ki/2.0_ki*abb13(13)
      abb13(13)=abb13(8)*abb13(13)
      abb13(19)=abb13(10)*abb13(7)
      abb13(23)=-abb13(11)+2.0_ki*abb13(19)
      abb13(23)=abb13(23)*abb13(7)
      abb13(23)=abb13(23)+abb13(16)
      abb13(23)=abb13(23)*spbl5l4
      abb13(24)=-abb13(16)+2.0_ki*abb13(12)
      abb13(24)=abb13(24)*abb13(20)*spbl4l3
      abb13(23)=abb13(23)+abb13(24)
      abb13(24)=abb13(16)*abb13(3)
      abb13(25)=abb13(12)*abb13(3)
      abb13(28)=abb13(24)-3.0_ki/2.0_ki*abb13(25)
      abb13(28)=abb13(28)*abb13(56)
      abb13(13)=abb13(28)-1.0_ki/2.0_ki*abb13(38)+abb13(13)-abb13(23)
      abb13(22)=abb13(36)-abb13(22)
      abb13(28)=-abb13(24)+2.0_ki*abb13(25)
      abb13(28)=abb13(28)*abb13(56)
      abb13(22)=abb13(28)+abb13(23)+1.0_ki/2.0_ki*abb13(22)
      abb13(23)=1.0_ki/2.0_ki*abb13(40)
      abb13(28)=1.0_ki/4.0_ki*spbl4k1
      abb13(36)=-abb13(18)*abb13(28)
      abb13(38)=-abb13(21)*abb13(28)
      abb13(39)=spak1l4*abb13(35)
      abb13(37)=-spak2l4*abb13(37)
      abb13(37)=abb13(39)+abb13(37)
      abb13(37)=1.0_ki/4.0_ki*abb13(37)
      abb13(17)=-abb13(17)*abb13(28)*abb13(59)
      abb13(14)=1.0_ki/4.0_ki*spbl5l3*abb13(14)
      abb13(39)=-abb13(7)*abb13(54)
      abb13(39)=-abb13(52)+abb13(39)
      abb13(39)=abb13(8)*abb13(39)
      abb13(11)=abb13(19)-abb13(11)
      abb13(11)=abb13(11)*abb13(7)
      abb13(19)=-spbl4k2*abb13(11)
      abb13(19)=abb13(19)+abb13(39)
      abb13(19)=1.0_ki/4.0_ki*abb13(19)
      abb13(39)=abb13(25)+abb13(24)
      abb13(40)=abb13(12)+abb13(16)
      abb13(27)=-abb13(40)*abb13(27)
      abb13(43)=-spbl5k1*abb13(39)
      abb13(47)=abb13(20)*abb13(24)
      abb13(48)=-spbl3k1*abb13(47)
      abb13(27)=abb13(48)+abb13(27)+abb13(43)
      abb13(27)=spak1l4*abb13(27)
      abb13(43)=-abb13(7)*abb13(49)
      abb13(43)=-abb13(44)+abb13(43)
      abb13(43)=abb13(8)*abb13(6)*abb13(3)*abb13(43)
      abb13(40)=abb13(40)*abb13(6)
      abb13(44)=spbl4k2*abb13(40)
      abb13(10)=abb13(10)*spbl4k2
      abb13(30)=-abb13(10)*abb13(30)
      abb13(30)=abb13(44)+abb13(30)
      abb13(30)=spak2l4*abb13(30)
      abb13(44)=abb13(46)+abb13(16)
      abb13(44)=spbl5k2*abb13(44)
      abb13(46)=-abb13(3)*abb13(55)
      abb13(32)=spak1l4*spbl5k2*abb13(32)
      abb13(32)=abb13(46)+abb13(32)
      abb13(32)=abb13(32)*abb13(29)
      abb13(16)=-abb13(12)+abb13(16)
      abb13(16)=abb13(16)*abb13(45)
      abb13(24)=-spbk2k1*spak1l3*abb13(24)*abb13(59)
      abb13(16)=abb13(24)+abb13(16)+abb13(30)+abb13(32)+abb13(43)+abb13(44)+abb&
      &13(27)
      abb13(16)=1.0_ki/4.0_ki*abb13(16)
      abb13(24)=abb13(58)*abb13(29)
      abb13(24)=abb13(24)-abb13(40)
      abb13(12)=abb13(12)*abb13(20)
      abb13(27)=spbl4k2*abb13(12)
      abb13(20)=-abb13(20)*abb13(33)
      abb13(20)=abb13(27)+abb13(20)
      abb13(20)=1.0_ki/4.0_ki*abb13(20)
      abb13(26)=-spbl4k2*abb13(26)
      abb13(18)=spbl5k2*abb13(18)
      abb13(10)=abb13(10)*abb13(31)
      abb13(21)=spbl3k2*abb13(21)
      abb13(10)=abb13(21)+abb13(10)+abb13(26)+abb13(18)
      abb13(18)=-spbl5l3*abb13(25)
      abb13(18)=abb13(18)+abb13(42)
      abb13(18)=spak1l3*abb13(18)
      abb13(10)=abb13(18)+3.0_ki*abb13(10)
      abb13(10)=1.0_ki/4.0_ki*abb13(10)
      abb13(18)=abb13(28)*abb13(11)
      abb13(11)=-spbl5k1*abb13(11)
      abb13(21)=spbl3k1*abb13(12)
      abb13(11)=abb13(11)+abb13(21)
      abb13(11)=1.0_ki/4.0_ki*abb13(11)
      abb13(12)=-abb13(28)*abb13(12)
      abb13(21)=-3.0_ki/4.0_ki*spak2l3*abb13(35)
      R2d13=abb13(41)
      rat2 = rat2 + R2d13
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='13' value='", &
          & R2d13, "'/>"
      end if
   end subroutine
end module p2_gg_httbar_abbrevd13h12
