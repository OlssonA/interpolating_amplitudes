module     p2_gg_httbar_abbrevd5h4
   use p2_gg_httbar_config, only: ki
   use p2_gg_httbar_kinematics, only: epstensor
   use p2_gg_httbar_globalsh4
   implicit none
   private
   complex(ki), dimension(61), public :: abb5
   complex(ki), public :: R2d5
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
      abb5(1)=sqrt(mT**2)
      abb5(2)=spak2l3**(-1)
      abb5(3)=spbl3k2**(-1)
      abb5(4)=spak2l4**(-1)
      abb5(5)=spbl5k2**(-1)
      abb5(6)=gs**4
      abb5(7)=abb5(6)*c3
      abb5(8)=mT*abb5(7)
      abb5(9)=i_*TR*e*gHT
      abb5(10)=abb5(9)*spae1e2
      abb5(11)=abb5(10)*abb5(8)*abb5(5)
      abb5(6)=abb5(9)*abb5(6)*NC
      abb5(12)=abb5(6)*mT
      abb5(13)=abb5(12)*abb5(5)
      abb5(14)=c2*abb5(13)
      abb5(15)=abb5(14)*spae1e2
      abb5(16)=abb5(11)-abb5(15)
      abb5(13)=c1*abb5(13)
      abb5(17)=abb5(13)*spae1e2
      abb5(16)=abb5(17)+1.0_ki/2.0_ki*abb5(16)
      abb5(18)=abb5(1)**2
      abb5(19)=-abb5(18)*abb5(16)
      abb5(20)=abb5(9)*abb5(7)*abb5(1)
      abb5(21)=abb5(6)*c2
      abb5(22)=abb5(21)*abb5(1)
      abb5(23)=abb5(20)-abb5(22)
      abb5(6)=abb5(6)*c1
      abb5(24)=abb5(6)*abb5(1)
      abb5(25)=abb5(24)+1.0_ki/2.0_ki*abb5(23)
      abb5(26)=spae1l5*spae2k2
      abb5(27)=mH**2*abb5(3)*abb5(2)
      abb5(28)=abb5(26)*abb5(27)
      abb5(29)=abb5(25)*abb5(28)
      abb5(19)=abb5(29)+abb5(19)
      abb5(29)=spbl4e2*spbk2e1
      abb5(19)=abb5(19)*abb5(29)
      abb5(11)=abb5(11)-abb5(17)
      abb5(17)=1.0_ki/2.0_ki*abb5(18)
      abb5(30)=abb5(17)*abb5(11)
      abb5(20)=abb5(20)-abb5(24)
      abb5(31)=-abb5(22)-1.0_ki/2.0_ki*abb5(20)
      abb5(32)=spae1k2*spae2l5
      abb5(33)=abb5(32)*abb5(27)
      abb5(34)=-abb5(31)*abb5(33)
      abb5(35)=abb5(18)*abb5(15)
      abb5(30)=abb5(34)+abb5(35)+abb5(30)
      abb5(34)=spbl4e1*spbk2e2
      abb5(30)=abb5(30)*abb5(34)
      abb5(35)=abb5(13)+abb5(14)
      abb5(36)=spae1e2*spbe2e1
      abb5(37)=abb5(18)*abb5(36)
      abb5(38)=abb5(37)*abb5(35)
      abb5(39)=abb5(24)+abb5(22)
      abb5(40)=1.0_ki/2.0_ki*abb5(36)
      abb5(41)=abb5(39)*abb5(40)
      abb5(10)=abb5(10)*spbe2e1
      abb5(42)=abb5(7)*abb5(10)
      abb5(43)=abb5(42)*abb5(1)
      abb5(41)=abb5(41)+abb5(43)
      abb5(43)=spak2l5*abb5(41)*abb5(27)
      abb5(44)=abb5(10)*abb5(8)
      abb5(45)=2.0_ki*abb5(44)
      abb5(46)=abb5(45)*abb5(5)
      abb5(47)=abb5(18)*abb5(46)
      abb5(38)=abb5(43)+abb5(47)+abb5(38)
      abb5(38)=spbl4k2*abb5(38)
      abb5(43)=abb5(9)*abb5(4)
      abb5(47)=mT**2*abb5(5)*abb5(1)
      abb5(48)=abb5(43)*abb5(7)*abb5(47)
      abb5(47)=abb5(47)*abb5(4)
      abb5(6)=abb5(47)*abb5(6)
      abb5(49)=abb5(48)-abb5(6)
      abb5(21)=abb5(47)*abb5(21)
      abb5(50)=-abb5(21)-1.0_ki/2.0_ki*abb5(49)
      abb5(51)=spae1k2*spbk2e2
      abb5(50)=abb5(50)*abb5(51)
      abb5(52)=spbl4e2*spae1l5
      abb5(25)=abb5(25)*abb5(52)
      abb5(25)=abb5(50)+abb5(25)
      abb5(25)=spbl3e1*spae2l3*abb5(25)
      abb5(48)=abb5(48)-abb5(21)
      abb5(50)=-1.0_ki/2.0_ki*abb5(48)-abb5(6)
      abb5(53)=spbk2e1*spae2k2
      abb5(50)=abb5(50)*abb5(53)
      abb5(54)=spbl4e1*spae2l5
      abb5(31)=-abb5(31)*abb5(54)
      abb5(31)=abb5(50)+abb5(31)
      abb5(31)=spae1l3*spbl3e2*abb5(31)
      abb5(12)=abb5(12)*abb5(4)
      abb5(50)=abb5(12)*c2
      abb5(55)=abb5(50)*spbe2e1
      abb5(56)=abb5(55)*abb5(18)
      abb5(9)=abb5(8)*abb5(9)
      abb5(57)=spbe2e1*abb5(4)*abb5(9)
      abb5(58)=-abb5(18)*abb5(57)
      abb5(58)=abb5(58)+abb5(56)
      abb5(12)=abb5(12)*c1
      abb5(59)=abb5(12)*spbe2e1
      abb5(60)=abb5(59)*abb5(18)
      abb5(58)=1.0_ki/2.0_ki*abb5(58)-abb5(60)
      abb5(58)=abb5(58)*abb5(26)
      abb5(60)=abb5(59)-abb5(57)
      abb5(17)=-abb5(17)*abb5(60)
      abb5(17)=abb5(56)+abb5(17)
      abb5(17)=abb5(17)*abb5(32)
      abb5(56)=abb5(50)+abb5(12)
      abb5(37)=-abb5(37)*abb5(56)
      abb5(45)=abb5(45)*abb5(4)
      abb5(18)=-abb5(18)*abb5(45)
      abb5(18)=abb5(18)+abb5(37)
      abb5(18)=spak2l5*abb5(18)
      abb5(37)=spbl4l3*spal3l5*abb5(41)
      abb5(41)=abb5(6)+abb5(21)
      abb5(61)=-abb5(40)*abb5(41)
      abb5(42)=-abb5(47)*abb5(42)
      abb5(42)=abb5(42)+abb5(61)
      abb5(42)=spak2l3*spbl3k2*abb5(42)
      abb5(17)=abb5(42)+abb5(37)+abb5(31)+abb5(25)+abb5(38)+abb5(18)+abb5(30)+a&
      &bb5(19)+abb5(58)+abb5(17)
      abb5(18)=abb5(35)*abb5(36)
      abb5(18)=abb5(18)+abb5(46)
      abb5(18)=abb5(18)*spbl4k2
      abb5(19)=abb5(56)*abb5(36)
      abb5(19)=abb5(19)+abb5(45)
      abb5(19)=abb5(19)*spak2l5
      abb5(18)=abb5(18)-abb5(19)
      abb5(19)=abb5(27)*abb5(18)
      abb5(25)=-abb5(36)*abb5(41)
      abb5(10)=2.0_ki*abb5(10)
      abb5(7)=abb5(10)*abb5(7)
      abb5(10)=-abb5(47)*abb5(7)
      abb5(10)=abb5(10)+abb5(25)
      abb5(10)=2.0_ki*abb5(10)+abb5(19)
      abb5(19)=-abb5(57)+abb5(55)
      abb5(19)=1.0_ki/2.0_ki*abb5(19)-abb5(59)
      abb5(19)=abb5(19)*abb5(26)
      abb5(25)=abb5(55)-1.0_ki/2.0_ki*abb5(60)
      abb5(25)=abb5(25)*abb5(32)
      abb5(16)=-abb5(16)*abb5(29)
      abb5(11)=abb5(15)+1.0_ki/2.0_ki*abb5(11)
      abb5(11)=abb5(11)*abb5(34)
      abb5(11)=abb5(11)+abb5(16)+abb5(19)+abb5(25)+abb5(18)
      abb5(15)=-2.0_ki*abb5(21)-abb5(49)
      abb5(15)=abb5(15)*abb5(51)
      abb5(16)=2.0_ki*abb5(24)+abb5(23)
      abb5(16)=abb5(16)*abb5(52)
      abb5(15)=abb5(15)+abb5(16)
      abb5(6)=-2.0_ki*abb5(6)-abb5(48)
      abb5(6)=abb5(6)*abb5(53)
      abb5(16)=2.0_ki*abb5(22)+abb5(20)
      abb5(16)=abb5(16)*abb5(54)
      abb5(6)=abb5(6)+abb5(16)
      abb5(9)=abb5(5)*abb5(9)
      abb5(16)=-abb5(9)+abb5(14)
      abb5(16)=1.0_ki/2.0_ki*abb5(16)-abb5(13)
      abb5(16)=spae1l3*abb5(16)*abb5(29)
      abb5(8)=abb5(8)*abb5(43)
      abb5(18)=abb5(8)-abb5(50)
      abb5(18)=abb5(12)+1.0_ki/2.0_ki*abb5(18)
      abb5(19)=-spbl3e1*abb5(18)*abb5(26)
      abb5(9)=abb5(9)-abb5(13)
      abb5(9)=-abb5(14)-1.0_ki/2.0_ki*abb5(9)
      abb5(9)=spae2l3*abb5(9)*abb5(34)
      abb5(8)=abb5(8)-abb5(12)
      abb5(8)=-abb5(50)-1.0_ki/2.0_ki*abb5(8)
      abb5(12)=spbl3e2*abb5(8)*abb5(32)
      abb5(13)=-spbk2e1*abb5(18)*abb5(28)
      abb5(8)=spbk2e2*abb5(8)*abb5(33)
      abb5(14)=abb5(36)*abb5(39)
      abb5(7)=abb5(1)*abb5(7)
      abb5(7)=abb5(7)+abb5(14)
      abb5(14)=abb5(56)*abb5(40)
      abb5(18)=abb5(44)*abb5(4)
      abb5(14)=abb5(14)+abb5(18)
      abb5(18)=-spak2l3*abb5(14)
      abb5(20)=abb5(35)*abb5(40)
      abb5(21)=abb5(44)*abb5(5)
      abb5(20)=abb5(20)+abb5(21)
      abb5(21)=spbl3k2*abb5(20)
      abb5(20)=spbl4l3*abb5(20)
      abb5(14)=-spal3l5*abb5(14)
      R2d5=0.0_ki
      rat2 = rat2 + R2d5
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='5' value='", &
          & R2d5, "'/>"
      end if
   end subroutine
end module p2_gg_httbar_abbrevd5h4
