module     p0_ubaru_httbar_abbrevd2h5
   use p0_ubaru_httbar_config, only: ki
   use p0_ubaru_httbar_kinematics, only: epstensor
   use p0_ubaru_httbar_globalsh5
   implicit none
   private
   complex(ki), dimension(37), public :: abb2
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
      abb2(3)=spak2l4**(-1)
      abb2(4)=sqrt(mT**2)
      abb2(5)=spak2l3**(-1)
      abb2(6)=spbl3k2**(-1)
      abb2(7)=spak2l5**(-1)
      abb2(8)=spbl5k2**(-1)
      abb2(9)=NC*c2
      abb2(9)=abb2(9)-c1
      abb2(9)=abb2(9)*i_*e*gHT*abb2(1)*TR**2*gs**4
      abb2(10)=-abb2(2)*abb2(9)
      abb2(11)=spak1l5*mT
      abb2(12)=-abb2(11)*abb2(10)
      abb2(13)=-spak1l5*abb2(10)
      abb2(14)=abb2(13)*abb2(4)
      abb2(15)=abb2(14)+abb2(12)
      abb2(16)=abb2(15)*spbl4k2
      abb2(17)=spak2l3*abb2(3)
      abb2(18)=abb2(17)*spbl3k2
      abb2(19)=abb2(18)*abb2(12)
      abb2(16)=abb2(16)+abb2(19)
      abb2(19)=abb2(9)*spak1l5
      abb2(20)=mT**3
      abb2(21)=abb2(20)*abb2(19)
      abb2(22)=mT**2
      abb2(23)=-abb2(22)*abb2(19)
      abb2(24)=-abb2(4)*abb2(23)
      abb2(24)=abb2(24)+abb2(21)
      abb2(25)=abb2(8)*spbl4k2
      abb2(24)=abb2(24)*abb2(25)
      abb2(26)=abb2(18)*abb2(8)
      abb2(27)=abb2(21)*abb2(26)
      abb2(24)=abb2(24)+abb2(27)
      abb2(24)=abb2(7)*abb2(24)
      abb2(27)=abb2(10)*abb2(4)
      abb2(28)=spak1l5**2
      abb2(29)=abb2(28)*abb2(27)
      abb2(30)=-mT*abb2(10)
      abb2(28)=-abb2(28)*abb2(30)
      abb2(29)=abb2(29)+abb2(28)
      abb2(31)=-spbl4k2*abb2(29)
      abb2(32)=-abb2(28)*abb2(18)
      abb2(31)=abb2(31)+abb2(32)
      abb2(31)=spbl5k1*abb2(31)
      abb2(24)=abb2(24)+abb2(31)
      abb2(31)=spak2l5*abb2(16)
      abb2(29)=spbl4k1*abb2(29)
      abb2(28)=abb2(28)*abb2(17)*spbl3k1
      abb2(28)=abb2(28)+abb2(29)+abb2(31)
      abb2(28)=spbl5k2*abb2(28)
      abb2(11)=-abb2(9)*abb2(11)
      abb2(29)=3.0_ki*abb2(4)
      abb2(31)=-abb2(11)*abb2(29)
      abb2(23)=abb2(31)-abb2(23)
      abb2(23)=abb2(4)*abb2(23)
      abb2(21)=-2.0_ki*abb2(21)+abb2(23)
      abb2(21)=abb2(3)*abb2(21)
      abb2(19)=abb2(19)*abb2(29)
      abb2(11)=2.0_ki*abb2(11)+abb2(19)
      abb2(19)=mH**2*abb2(6)*abb2(5)
      abb2(23)=abb2(19)*spbl4k2
      abb2(11)=abb2(11)*abb2(23)
      abb2(31)=-2.0_ki*abb2(12)+3.0_ki*abb2(14)
      abb2(32)=spak1l3*spbl4l3
      abb2(33)=abb2(32)*spbk2k1
      abb2(31)=abb2(31)*abb2(33)
      abb2(11)=abb2(31)+abb2(21)+abb2(11)+abb2(28)+2.0_ki*abb2(24)
      abb2(21)=2.0_ki*abb2(16)
      abb2(13)=-abb2(22)*abb2(13)
      abb2(24)=-abb2(4)*abb2(12)
      abb2(13)=abb2(24)+abb2(13)
      abb2(13)=abb2(3)*abb2(4)*abb2(13)
      abb2(24)=-abb2(14)*abb2(23)
      abb2(13)=abb2(24)+abb2(13)-abb2(16)
      abb2(13)=2.0_ki*abb2(13)
      abb2(24)=4.0_ki*abb2(16)
      abb2(28)=spbl5k2*abb2(15)
      abb2(12)=abb2(12)*abb2(17)
      abb2(31)=spbl5k2*abb2(12)
      abb2(34)=-abb2(22)*abb2(10)
      abb2(35)=abb2(34)*abb2(4)
      abb2(10)=-abb2(20)*abb2(10)
      abb2(35)=abb2(35)+abb2(10)
      abb2(36)=abb2(35)*abb2(25)
      abb2(26)=abb2(10)*abb2(26)
      abb2(26)=abb2(36)+abb2(26)
      abb2(26)=abb2(7)*abb2(26)
      abb2(36)=abb2(30)*abb2(4)**2
      abb2(36)=abb2(36)-abb2(10)
      abb2(36)=abb2(3)*abb2(36)
      abb2(37)=abb2(27)+abb2(30)
      abb2(23)=-abb2(37)*abb2(23)
      abb2(23)=abb2(26)+abb2(36)+abb2(23)
      abb2(23)=spak1k2*abb2(23)
      abb2(26)=-abb2(37)*abb2(32)
      abb2(23)=abb2(26)+abb2(23)
      abb2(14)=-spbl4l3*abb2(14)
      abb2(26)=-abb2(30)*abb2(29)
      abb2(26)=abb2(26)-abb2(34)
      abb2(26)=abb2(4)*abb2(26)
      abb2(32)=2.0_ki*abb2(10)
      abb2(26)=abb2(32)+abb2(26)
      abb2(26)=spbl4k2*abb2(26)
      abb2(29)=-abb2(34)*abb2(29)
      abb2(29)=abb2(32)+abb2(29)
      abb2(29)=abb2(29)*abb2(18)
      abb2(22)=-abb2(4)*abb2(22)
      abb2(20)=abb2(22)-abb2(20)
      abb2(20)=abb2(3)*abb2(9)*abb2(20)
      abb2(22)=2.0_ki*abb2(30)
      abb2(32)=-abb2(22)*abb2(33)
      abb2(20)=abb2(32)+abb2(29)+abb2(26)+2.0_ki*abb2(20)
      abb2(20)=abb2(8)*abb2(20)
      abb2(26)=abb2(4)*abb2(22)
      abb2(26)=abb2(26)+abb2(34)
      abb2(26)=abb2(4)*abb2(26)
      abb2(10)=abb2(26)-abb2(10)
      abb2(10)=abb2(3)*abb2(10)
      abb2(26)=abb2(27)-abb2(30)
      abb2(29)=-spbl4k2*abb2(26)
      abb2(18)=abb2(30)*abb2(18)
      abb2(10)=abb2(18)+abb2(29)+abb2(10)
      abb2(10)=spak2l5*abb2(10)
      abb2(9)=-abb2(25)*mT*abb2(9)
      abb2(18)=abb2(30)+2.0_ki*abb2(27)
      abb2(27)=-spak2l5*spbl4k2*abb2(18)
      abb2(9)=abb2(9)+abb2(27)
      abb2(9)=abb2(9)*abb2(19)
      abb2(18)=spal3l5*spbl4l3*abb2(18)
      abb2(9)=-abb2(18)+abb2(10)+abb2(9)
      abb2(10)=-spbl4k1*abb2(15)
      abb2(12)=-spbl3k1*abb2(12)
      abb2(9)=abb2(12)+abb2(10)+abb2(20)+2.0_ki*abb2(9)
      abb2(10)=-abb2(8)*abb2(3)*abb2(35)
      abb2(12)=-abb2(30)*abb2(19)*abb2(25)
      abb2(10)=abb2(10)+abb2(12)
      abb2(10)=4.0_ki*abb2(10)
      abb2(12)=-2.0_ki*abb2(26)
      abb2(15)=abb2(17)*abb2(22)
      abb2(17)=-abb2(8)*abb2(22)*spbl4l3
      R2d2=-abb2(16)
      rat2 = rat2 + R2d2
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='2' value='", &
          & R2d2, "'/>"
      end if
   end subroutine
end module p0_ubaru_httbar_abbrevd2h5
