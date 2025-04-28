module     p0_ubaru_httbar_abbrevd4h5
   use p0_ubaru_httbar_config, only: ki
   use p0_ubaru_httbar_kinematics, only: epstensor
   use p0_ubaru_httbar_globalsh5
   implicit none
   private
   complex(ki), dimension(49), public :: abb4
   complex(ki), public :: R2d4
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
      abb4(1)=1.0_ki/(mH**2+mT**2-es34-es45+es12)
      abb4(2)=es12**(-1)
      abb4(3)=spbl5k2**(-1)
      abb4(4)=spak2l4**(-1)
      abb4(5)=sqrt(mT**2)
      abb4(6)=spak2l3**(-1)
      abb4(7)=spbl3k2**(-1)
      abb4(8)=spbl4k2**(-1)
      abb4(9)=NC*c2
      abb4(9)=abb4(9)-c1
      abb4(10)=i_*e*gHT*abb4(1)*TR**2*gs**4
      abb4(11)=abb4(10)*abb4(2)
      abb4(12)=-abb4(11)*abb4(9)
      abb4(13)=-spbl4k2*abb4(12)
      abb4(14)=abb4(13)*abb4(5)
      abb4(15)=-spbl4k2*abb4(9)
      abb4(11)=abb4(11)*mT
      abb4(16)=-abb4(11)*abb4(15)
      abb4(17)=abb4(14)+abb4(16)
      abb4(18)=abb4(17)*spak1l5
      abb4(19)=spbl3k2*abb4(3)
      abb4(20)=abb4(19)*spak1l3
      abb4(21)=abb4(20)*abb4(16)
      abb4(18)=abb4(18)+abb4(21)
      abb4(11)=-abb4(11)*abb4(9)
      abb4(21)=abb4(11)*abb4(4)
      abb4(22)=spal3l5*spak1k2
      abb4(23)=abb4(22)*spbl3k2
      abb4(24)=abb4(21)*abb4(23)
      abb4(25)=abb4(18)+abb4(24)
      abb4(26)=abb4(10)*abb4(9)
      abb4(27)=mT**3
      abb4(28)=-abb4(27)*abb4(26)
      abb4(29)=abb4(28)*abb4(4)
      abb4(30)=mT**2
      abb4(26)=-abb4(30)*abb4(26)
      abb4(31)=abb4(5)*abb4(4)
      abb4(32)=-abb4(31)*abb4(26)
      abb4(32)=-abb4(29)+abb4(32)
      abb4(32)=abb4(3)*spbl4k2*abb4(32)
      abb4(33)=abb4(12)*abb4(5)
      abb4(34)=spbl4k2**2
      abb4(35)=abb4(34)*abb4(33)
      abb4(34)=-abb4(34)*abb4(11)
      abb4(35)=abb4(35)-abb4(34)
      abb4(36)=spak2l5*abb4(35)
      abb4(37)=2.0_ki*abb4(4)
      abb4(38)=abb4(11)*abb4(37)
      abb4(39)=abb4(23)*abb4(38)
      abb4(40)=abb4(39)+abb4(18)
      abb4(40)=spbl4k1*abb4(40)
      abb4(41)=spak2l5*abb4(6)*abb4(7)*mH**2
      abb4(42)=abb4(41)*abb4(4)
      abb4(10)=abb4(10)*mT
      abb4(15)=-abb4(42)*abb4(10)*abb4(15)
      abb4(43)=abb4(19)*spak2l3
      abb4(44)=-abb4(34)*abb4(43)
      abb4(22)=abb4(22)*abb4(4)
      abb4(45)=abb4(22)*abb4(16)
      abb4(46)=spbl3k1*abb4(45)
      abb4(15)=abb4(40)+abb4(46)+abb4(44)+abb4(15)+abb4(36)+abb4(32)
      abb4(15)=spak1l4*abb4(15)
      abb4(32)=3.0_ki*abb4(14)
      abb4(36)=abb4(8)*abb4(4)**2
      abb4(28)=abb4(28)*abb4(36)
      abb4(28)=2.0_ki*abb4(28)-abb4(32)+abb4(16)
      abb4(28)=abb4(23)*abb4(28)
      abb4(40)=-abb4(26)*abb4(37)
      abb4(9)=abb4(9)*abb4(10)
      abb4(10)=3.0_ki*abb4(31)
      abb4(44)=abb4(9)*abb4(10)
      abb4(40)=abb4(40)+abb4(44)
      abb4(40)=abb4(5)*abb4(40)
      abb4(40)=abb4(29)+abb4(40)
      abb4(40)=spak1l5*abb4(40)
      abb4(10)=-abb4(26)*abb4(10)
      abb4(10)=abb4(29)+abb4(10)
      abb4(10)=abb4(10)*abb4(20)
      abb4(35)=-spak1l5*abb4(35)
      abb4(34)=abb4(34)*abb4(20)
      abb4(34)=abb4(35)+abb4(34)
      abb4(34)=spak2l4*abb4(34)
      abb4(10)=2.0_ki*abb4(34)+abb4(10)+abb4(40)+abb4(28)+abb4(15)
      abb4(15)=2.0_ki*abb4(18)
      abb4(28)=abb4(31)*abb4(11)
      abb4(34)=-abb4(30)*abb4(12)
      abb4(35)=abb4(4)*abb4(34)
      abb4(35)=-abb4(28)+2.0_ki*abb4(13)+abb4(35)
      abb4(35)=abb4(5)*abb4(35)
      abb4(40)=2.0_ki*abb4(16)
      abb4(35)=abb4(40)+abb4(35)
      abb4(35)=spak1l5*abb4(35)
      abb4(44)=abb4(34)*abb4(31)
      abb4(46)=abb4(40)+abb4(44)
      abb4(46)=abb4(46)*abb4(20)
      abb4(12)=-abb4(27)*abb4(12)
      abb4(23)=-abb4(12)*abb4(23)*abb4(36)
      abb4(23)=abb4(23)+abb4(46)+abb4(35)-abb4(24)
      abb4(23)=2.0_ki*abb4(23)
      abb4(18)=abb4(39)-abb4(18)
      abb4(18)=2.0_ki*abb4(18)
      abb4(24)=2.0_ki*abb4(25)
      abb4(17)=spak1l4*abb4(17)
      abb4(35)=abb4(12)*abb4(4)
      abb4(36)=abb4(5)**2*abb4(21)
      abb4(36)=abb4(35)+abb4(36)
      abb4(36)=spak1k2*abb4(36)
      abb4(27)=abb4(13)*abb4(27)
      abb4(39)=-abb4(4)*abb4(27)
      abb4(30)=abb4(13)*abb4(30)
      abb4(31)=-abb4(30)*abb4(31)
      abb4(31)=abb4(39)+abb4(31)
      abb4(31)=abb4(3)*abb4(31)
      abb4(39)=-abb4(16)*abb4(42)
      abb4(31)=abb4(31)+abb4(39)
      abb4(31)=spak1k2*abb4(31)
      abb4(39)=spak1l4*abb4(16)*abb4(19)
      abb4(42)=abb4(35)-abb4(44)
      abb4(42)=spak1k2*abb4(42)*abb4(19)
      abb4(46)=spal3l5*spbl3k2
      abb4(47)=spak1l4*abb4(21)*abb4(46)
      abb4(46)=-abb4(33)*abb4(46)
      abb4(14)=-spal3l5*abb4(14)
      abb4(48)=abb4(33)+abb4(11)
      abb4(49)=-spak1l5*abb4(48)
      abb4(20)=-abb4(11)*abb4(20)
      abb4(20)=abb4(49)+abb4(20)
      abb4(20)=spbl4k1*abb4(20)
      abb4(33)=-2.0_ki*abb4(33)+abb4(11)
      abb4(33)=spbl4l3*spal3l5*abb4(33)
      abb4(20)=abb4(20)+abb4(33)
      abb4(26)=-abb4(4)*abb4(26)
      abb4(33)=abb4(5)*abb4(16)
      abb4(26)=3.0_ki*abb4(33)+abb4(26)+abb4(30)
      abb4(26)=abb4(5)*abb4(26)
      abb4(26)=abb4(26)-2.0_ki*abb4(27)-abb4(29)
      abb4(26)=abb4(3)*abb4(26)
      abb4(27)=-abb4(34)*abb4(37)
      abb4(13)=4.0_ki*abb4(28)+abb4(27)-abb4(13)
      abb4(13)=abb4(5)*abb4(13)
      abb4(12)=abb4(12)*abb4(37)
      abb4(12)=abb4(12)-abb4(16)
      abb4(13)=abb4(13)+abb4(12)
      abb4(13)=spak2l5*abb4(13)
      abb4(9)=abb4(4)*abb4(9)
      abb4(9)=abb4(32)-abb4(40)+abb4(9)
      abb4(9)=abb4(9)*abb4(41)
      abb4(12)=-4.0_ki*abb4(44)+abb4(12)
      abb4(12)=abb4(12)*abb4(43)
      abb4(16)=-abb4(11)*abb4(22)*spbl3k1
      abb4(9)=abb4(16)+abb4(12)+abb4(9)+abb4(26)+abb4(13)+2.0_ki*abb4(20)
      abb4(12)=-abb4(35)-abb4(44)
      abb4(12)=abb4(3)*abb4(12)
      abb4(13)=abb4(21)*abb4(41)
      abb4(12)=abb4(12)+abb4(13)
      abb4(12)=4.0_ki*abb4(12)
      abb4(13)=-2.0_ki*abb4(48)
      abb4(11)=-2.0_ki*abb4(11)*abb4(19)
      abb4(16)=spal3l5*abb4(38)
      R2d4=-abb4(25)
      rat2 = rat2 + R2d4
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='4' value='", &
          & R2d4, "'/>"
      end if
   end subroutine
end module p0_ubaru_httbar_abbrevd4h5
