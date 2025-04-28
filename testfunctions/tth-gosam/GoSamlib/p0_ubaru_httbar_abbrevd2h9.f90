module     p0_ubaru_httbar_abbrevd2h9
   use p0_ubaru_httbar_config, only: ki
   use p0_ubaru_httbar_kinematics, only: epstensor
   use p0_ubaru_httbar_globalsh9
   implicit none
   private
   complex(ki), dimension(49), public :: abb2
   complex(ki), public :: R2d2
   public :: init_abbrev
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
contains
   subroutine     init_abbrev()
      use p0_ubaru_httbar_config, only: deltaOS, &
     &    logfile, debug_nlo_diagrams
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_model
      use p0_ubaru_httbar_color, only: TR
      use p0_ubaru_httbar_globalsl1, only: epspow
      implicit none
      abb2(1)=1.0_ki/(-mT**2+es34)
      abb2(2)=es12**(-1)
      abb2(3)=spbl4k2**(-1)
      abb2(4)=spak2l5**(-1)
      abb2(5)=sqrt(mT**2)
      abb2(6)=spak2l3**(-1)
      abb2(7)=spbl3k2**(-1)
      abb2(8)=spbl5k2**(-1)
      abb2(9)=NC*c2
      abb2(9)=abb2(9)-c1
      abb2(10)=i_*e*gHT*abb2(1)*TR**2*gs**4
      abb2(11)=abb2(10)*abb2(2)
      abb2(12)=-abb2(11)*abb2(9)
      abb2(13)=-spbl5k2*abb2(12)
      abb2(14)=abb2(13)*abb2(5)
      abb2(15)=-spbl5k2*abb2(9)
      abb2(11)=abb2(11)*mT
      abb2(16)=-abb2(11)*abb2(15)
      abb2(17)=abb2(14)+abb2(16)
      abb2(18)=abb2(17)*spak1l4
      abb2(19)=spbl3k2*abb2(3)
      abb2(20)=abb2(19)*spak1l3
      abb2(21)=abb2(20)*abb2(16)
      abb2(18)=abb2(18)+abb2(21)
      abb2(11)=-abb2(11)*abb2(9)
      abb2(21)=abb2(11)*abb2(4)
      abb2(22)=spal3l4*spak1k2
      abb2(23)=abb2(22)*spbl3k2
      abb2(24)=abb2(21)*abb2(23)
      abb2(25)=abb2(18)+abb2(24)
      abb2(26)=abb2(10)*abb2(9)
      abb2(27)=mT**3
      abb2(28)=-abb2(27)*abb2(26)
      abb2(29)=abb2(28)*abb2(4)
      abb2(30)=mT**2
      abb2(26)=-abb2(30)*abb2(26)
      abb2(31)=abb2(5)*abb2(4)
      abb2(32)=-abb2(31)*abb2(26)
      abb2(32)=-abb2(29)+abb2(32)
      abb2(32)=abb2(3)*spbl5k2*abb2(32)
      abb2(33)=abb2(12)*abb2(5)
      abb2(34)=spbl5k2**2
      abb2(35)=abb2(34)*abb2(33)
      abb2(34)=-abb2(34)*abb2(11)
      abb2(35)=abb2(35)-abb2(34)
      abb2(36)=spak2l4*abb2(35)
      abb2(37)=2.0_ki*abb2(4)
      abb2(38)=abb2(11)*abb2(37)
      abb2(39)=abb2(23)*abb2(38)
      abb2(40)=abb2(39)+abb2(18)
      abb2(40)=spbl5k1*abb2(40)
      abb2(41)=spak2l4*abb2(6)*abb2(7)*mH**2
      abb2(42)=abb2(41)*abb2(4)
      abb2(10)=abb2(10)*mT
      abb2(15)=-abb2(42)*abb2(10)*abb2(15)
      abb2(43)=abb2(19)*spak2l3
      abb2(44)=-abb2(34)*abb2(43)
      abb2(22)=abb2(22)*abb2(4)
      abb2(45)=abb2(22)*abb2(16)
      abb2(46)=spbl3k1*abb2(45)
      abb2(15)=abb2(40)+abb2(46)+abb2(44)+abb2(15)+abb2(36)+abb2(32)
      abb2(15)=spak1l5*abb2(15)
      abb2(32)=3.0_ki*abb2(14)
      abb2(36)=abb2(8)*abb2(4)**2
      abb2(28)=abb2(28)*abb2(36)
      abb2(28)=2.0_ki*abb2(28)-abb2(32)+abb2(16)
      abb2(28)=abb2(23)*abb2(28)
      abb2(40)=-abb2(26)*abb2(37)
      abb2(9)=abb2(9)*abb2(10)
      abb2(10)=3.0_ki*abb2(31)
      abb2(44)=abb2(9)*abb2(10)
      abb2(40)=abb2(40)+abb2(44)
      abb2(40)=abb2(5)*abb2(40)
      abb2(40)=abb2(29)+abb2(40)
      abb2(40)=spak1l4*abb2(40)
      abb2(10)=-abb2(26)*abb2(10)
      abb2(10)=abb2(29)+abb2(10)
      abb2(10)=abb2(10)*abb2(20)
      abb2(35)=-spak1l4*abb2(35)
      abb2(34)=abb2(34)*abb2(20)
      abb2(34)=abb2(35)+abb2(34)
      abb2(34)=spak2l5*abb2(34)
      abb2(10)=2.0_ki*abb2(34)+abb2(10)+abb2(40)+abb2(28)+abb2(15)
      abb2(15)=2.0_ki*abb2(18)
      abb2(28)=abb2(31)*abb2(11)
      abb2(34)=-abb2(30)*abb2(12)
      abb2(35)=abb2(4)*abb2(34)
      abb2(35)=-abb2(28)+2.0_ki*abb2(13)+abb2(35)
      abb2(35)=abb2(5)*abb2(35)
      abb2(40)=2.0_ki*abb2(16)
      abb2(35)=abb2(40)+abb2(35)
      abb2(35)=spak1l4*abb2(35)
      abb2(44)=abb2(34)*abb2(31)
      abb2(46)=abb2(40)+abb2(44)
      abb2(46)=abb2(46)*abb2(20)
      abb2(12)=-abb2(27)*abb2(12)
      abb2(23)=-abb2(12)*abb2(23)*abb2(36)
      abb2(23)=abb2(23)+abb2(46)+abb2(35)-abb2(24)
      abb2(23)=2.0_ki*abb2(23)
      abb2(18)=abb2(39)-abb2(18)
      abb2(18)=2.0_ki*abb2(18)
      abb2(24)=2.0_ki*abb2(25)
      abb2(27)=abb2(13)*abb2(27)
      abb2(35)=-abb2(4)*abb2(27)
      abb2(30)=abb2(13)*abb2(30)
      abb2(31)=-abb2(30)*abb2(31)
      abb2(31)=abb2(35)+abb2(31)
      abb2(31)=abb2(3)*abb2(31)
      abb2(35)=-abb2(16)*abb2(42)
      abb2(31)=abb2(31)+abb2(35)
      abb2(31)=spak1k2*abb2(31)
      abb2(17)=spak1l5*abb2(17)
      abb2(35)=abb2(12)*abb2(4)
      abb2(36)=abb2(5)**2*abb2(21)
      abb2(36)=abb2(35)+abb2(36)
      abb2(36)=spak1k2*abb2(36)
      abb2(39)=spak1l5*abb2(16)*abb2(19)
      abb2(42)=abb2(35)-abb2(44)
      abb2(42)=spak1k2*abb2(42)*abb2(19)
      abb2(46)=spal3l4*spbl3k2
      abb2(47)=spak1l5*abb2(21)*abb2(46)
      abb2(46)=-abb2(33)*abb2(46)
      abb2(14)=-spal3l4*abb2(14)
      abb2(48)=abb2(33)+abb2(11)
      abb2(49)=-spak1l4*abb2(48)
      abb2(20)=-abb2(11)*abb2(20)
      abb2(20)=abb2(49)+abb2(20)
      abb2(20)=spbl5k1*abb2(20)
      abb2(33)=-2.0_ki*abb2(33)+abb2(11)
      abb2(33)=spbl5l3*spal3l4*abb2(33)
      abb2(20)=abb2(20)+abb2(33)
      abb2(26)=-abb2(4)*abb2(26)
      abb2(33)=abb2(5)*abb2(16)
      abb2(26)=3.0_ki*abb2(33)+abb2(26)+abb2(30)
      abb2(26)=abb2(5)*abb2(26)
      abb2(26)=abb2(26)-2.0_ki*abb2(27)-abb2(29)
      abb2(26)=abb2(3)*abb2(26)
      abb2(27)=-abb2(34)*abb2(37)
      abb2(13)=4.0_ki*abb2(28)+abb2(27)-abb2(13)
      abb2(13)=abb2(5)*abb2(13)
      abb2(12)=abb2(12)*abb2(37)
      abb2(12)=abb2(12)-abb2(16)
      abb2(13)=abb2(13)+abb2(12)
      abb2(13)=spak2l4*abb2(13)
      abb2(9)=abb2(4)*abb2(9)
      abb2(9)=abb2(32)-abb2(40)+abb2(9)
      abb2(9)=abb2(9)*abb2(41)
      abb2(12)=-4.0_ki*abb2(44)+abb2(12)
      abb2(12)=abb2(12)*abb2(43)
      abb2(16)=-abb2(11)*abb2(22)*spbl3k1
      abb2(9)=abb2(16)+abb2(12)+abb2(9)+abb2(26)+abb2(13)+2.0_ki*abb2(20)
      abb2(12)=-abb2(35)-abb2(44)
      abb2(12)=abb2(3)*abb2(12)
      abb2(13)=abb2(21)*abb2(41)
      abb2(12)=abb2(12)+abb2(13)
      abb2(12)=4.0_ki*abb2(12)
      abb2(13)=-2.0_ki*abb2(48)
      abb2(11)=-2.0_ki*abb2(11)*abb2(19)
      abb2(16)=spal3l4*abb2(38)
      R2d2=-abb2(25)
      rat2 = rat2 + R2d2
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='2' value='", &
          & R2d2, "'/>"
      end if
   end subroutine
end module p0_ubaru_httbar_abbrevd2h9
