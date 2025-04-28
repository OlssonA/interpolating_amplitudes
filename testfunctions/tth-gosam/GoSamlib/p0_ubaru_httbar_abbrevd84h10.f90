module     p0_ubaru_httbar_abbrevd84h10
   use p0_ubaru_httbar_config, only: ki
   use p0_ubaru_httbar_kinematics, only: epstensor
   use p0_ubaru_httbar_globalsh10
   implicit none
   private
   complex(ki), dimension(28), public :: abb84
   complex(ki), public :: R2d84
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
      abb84(1)=sqrt(mT**2)
      abb84(2)=NC**(-1)
      abb84(3)=spbl4k2**(-1)
      abb84(4)=spak2l5**(-1)
      abb84(5)=spbl5k2**(-1)
      abb84(6)=abb84(2)*c1
      abb84(7)=abb84(6)*abb84(1)
      abb84(8)=2.0_ki*c2
      abb84(9)=abb84(8)*abb84(1)
      abb84(7)=abb84(7)-abb84(9)
      abb84(7)=abb84(7)*abb84(2)
      abb84(10)=-spak2l4*abb84(7)
      abb84(11)=c1*spak2l4
      abb84(12)=abb84(11)*abb84(1)
      abb84(10)=abb84(10)-abb84(12)
      abb84(12)=TR**2*gHT*e*i_*gs**4
      abb84(13)=abb84(12)*spbl5k1
      abb84(14)=4.0_ki*abb84(13)
      abb84(15)=-abb84(10)*abb84(14)
      abb84(16)=c1*abb84(1)
      abb84(7)=abb84(7)+abb84(16)
      abb84(16)=abb84(3)*spbl4k1
      abb84(17)=spak2l4*abb84(4)
      abb84(18)=-abb84(7)*abb84(16)*abb84(17)
      abb84(9)=-abb84(3)*abb84(9)
      abb84(19)=abb84(17)*abb84(5)
      abb84(20)=abb84(19)*abb84(1)
      abb84(21)=c2*abb84(20)
      abb84(9)=abb84(9)+abb84(21)
      abb84(21)=abb84(1)*abb84(3)
      abb84(20)=-abb84(20)+2.0_ki*abb84(21)
      abb84(21)=abb84(20)*abb84(6)
      abb84(9)=2.0_ki*abb84(9)+abb84(21)
      abb84(9)=abb84(2)*abb84(9)
      abb84(20)=c1*abb84(20)
      abb84(9)=abb84(20)+abb84(9)
      abb84(9)=spbl5k1*abb84(9)
      abb84(6)=abb84(8)-abb84(6)
      abb84(8)=abb84(19)-abb84(3)
      abb84(19)=-abb84(2)*abb84(8)*abb84(6)
      abb84(8)=c1*abb84(8)
      abb84(8)=abb84(8)+abb84(19)
      abb84(19)=mT*spbl5k1
      abb84(8)=abb84(8)*abb84(19)
      abb84(8)=abb84(8)+abb84(9)+abb84(18)
      abb84(8)=mT*abb84(8)
      abb84(9)=abb84(1)**2
      abb84(18)=abb84(9)*abb84(3)
      abb84(18)=abb84(18)-spak2l4
      abb84(20)=abb84(2)*abb84(18)*abb84(6)
      abb84(18)=-c1*abb84(18)
      abb84(18)=abb84(18)+abb84(20)
      abb84(18)=spbl5k1*abb84(18)
      abb84(8)=abb84(18)+abb84(8)
      abb84(8)=mT*abb84(8)
      abb84(18)=spbl5k1*abb84(10)
      abb84(8)=abb84(18)+abb84(8)
      abb84(18)=4.0_ki*abb84(12)
      abb84(8)=abb84(8)*abb84(18)
      abb84(20)=-abb84(3)*abb84(6)
      abb84(21)=abb84(20)*abb84(2)
      abb84(22)=c1*abb84(3)
      abb84(21)=abb84(21)+abb84(22)
      abb84(13)=-8.0_ki*mT**3*abb84(13)*abb84(21)*abb84(4)*abb84(5)
      abb84(22)=2.0_ki*spbl5k1
      abb84(10)=abb84(10)*abb84(22)
      abb84(22)=abb84(6)*abb84(2)
      abb84(23)=-spak2l4*abb84(22)
      abb84(23)=abb84(11)+abb84(23)
      abb84(23)=abb84(23)*abb84(19)
      abb84(10)=abb84(10)+abb84(23)
      abb84(10)=abb84(10)*abb84(18)
      abb84(14)=-mT*abb84(21)*abb84(14)
      abb84(20)=-abb84(2)*spak2l4*abb84(20)
      abb84(23)=-abb84(3)*abb84(11)
      abb84(23)=abb84(23)+abb84(20)
      abb84(12)=2.0_ki*abb84(12)
      abb84(19)=abb84(19)*abb84(12)
      abb84(23)=abb84(19)*spbl5l4*abb84(23)
      abb84(24)=abb84(12)*spbl5k1
      abb84(25)=-abb84(24)*spak2l5*abb84(7)
      abb84(26)=-abb84(24)*spak1k2*abb84(7)
      abb84(24)=abb84(24)*spal4l5*abb84(7)
      abb84(27)=abb84(21)*abb84(19)
      abb84(6)=-abb84(2)*abb84(17)*abb84(6)
      abb84(28)=abb84(17)*c1
      abb84(6)=abb84(6)+abb84(28)
      abb84(28)=-spbl5k1*spal4l5*abb84(6)
      abb84(22)=c1-abb84(22)
      abb84(17)=mT**2*abb84(22)*spbk2k1*abb84(5)*abb84(17)**2
      abb84(17)=abb84(28)+abb84(17)
      abb84(22)=abb84(12)*mT
      abb84(17)=abb84(17)*abb84(22)
      abb84(22)=abb84(6)*abb84(22)
      abb84(28)=spbl5k1*spak1l4*abb84(7)
      abb84(9)=mT*abb84(9)*abb84(6)
      abb84(9)=abb84(28)+abb84(9)
      abb84(9)=abb84(9)*abb84(12)
      abb84(7)=abb84(7)*abb84(3)*abb84(4)
      abb84(12)=-mT*abb84(4)*abb84(21)
      abb84(7)=2.0_ki*abb84(7)+abb84(12)
      abb84(7)=mT*abb84(7)
      abb84(6)=abb84(7)+abb84(6)
      abb84(6)=abb84(6)*mT*abb84(18)
      abb84(7)=-abb84(11)*abb84(16)
      abb84(11)=spbl4k1*abb84(20)
      abb84(7)=abb84(7)+abb84(11)
      abb84(7)=abb84(7)*abb84(19)
      R2d84=0.0_ki
      rat2 = rat2 + R2d84
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='84' value='", &
          & R2d84, "'/>"
      end if
   end subroutine
end module p0_ubaru_httbar_abbrevd84h10
