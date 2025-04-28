module     p0_ubaru_httbar_abbrevd84h13
   use p0_ubaru_httbar_config, only: ki
   use p0_ubaru_httbar_kinematics, only: epstensor
   use p0_ubaru_httbar_globalsh13
   implicit none
   private
   complex(ki), dimension(31), public :: abb84
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
      abb84(3)=spak2l5**(-1)
      abb84(4)=spbl5k2**(-1)
      abb84(5)=spak2l4**(-1)
      abb84(6)=2.0_ki*c2
      abb84(7)=abb84(2)*c1
      abb84(7)=abb84(6)-abb84(7)
      abb84(8)=-abb84(2)*abb84(7)*spbl4k2
      abb84(9)=c1*spbl4k2
      abb84(8)=abb84(8)+abb84(9)
      abb84(10)=abb84(3)*spak1k2
      abb84(11)=abb84(10)*abb84(1)
      abb84(12)=-abb84(11)*abb84(8)
      abb84(13)=TR**2*gHT*e*i_*gs**4
      abb84(14)=abb84(13)*mT
      abb84(14)=4.0_ki*abb84(14)
      abb84(15)=-abb84(12)*abb84(14)
      abb84(16)=abb84(1)*abb84(5)
      abb84(17)=spak1k2*abb84(16)
      abb84(18)=abb84(4)*abb84(11)*spbl4k2
      abb84(17)=abb84(17)-abb84(18)
      abb84(18)=-abb84(3)*c2*abb84(17)
      abb84(19)=c1*abb84(3)
      abb84(17)=abb84(17)*abb84(19)
      abb84(20)=abb84(2)*abb84(17)
      abb84(18)=2.0_ki*abb84(18)+abb84(20)
      abb84(18)=abb84(2)*abb84(18)
      abb84(20)=abb84(4)*abb84(3)**2
      abb84(21)=spak1k2*abb84(6)*abb84(20)
      abb84(20)=c1*spak1k2*abb84(20)
      abb84(22)=abb84(20)*abb84(2)
      abb84(21)=abb84(21)-abb84(22)
      abb84(22)=-abb84(2)*spbl4k2*abb84(21)
      abb84(23)=abb84(20)*spbl4k2
      abb84(22)=abb84(23)+abb84(22)
      abb84(22)=mT*abb84(22)
      abb84(17)=abb84(22)+abb84(17)+abb84(18)
      abb84(17)=mT*abb84(17)
      abb84(18)=abb84(10)*abb84(5)
      abb84(22)=-abb84(2)*abb84(18)*abb84(7)
      abb84(23)=abb84(18)*c1
      abb84(22)=abb84(22)+abb84(23)
      abb84(23)=abb84(1)**2
      abb84(24)=-abb84(23)*abb84(22)
      abb84(17)=abb84(17)+abb84(24)
      abb84(17)=mT*abb84(17)
      abb84(17)=abb84(17)+abb84(12)
      abb84(17)=abb84(17)*abb84(14)
      abb84(21)=abb84(2)*abb84(5)*abb84(21)
      abb84(20)=abb84(20)*abb84(5)
      abb84(20)=-abb84(20)+abb84(21)
      abb84(20)=8.0_ki*abb84(20)*mT**4*abb84(13)
      abb84(21)=abb84(7)*abb84(2)
      abb84(24)=-spbl4k2*abb84(21)
      abb84(9)=abb84(9)+abb84(24)
      abb84(9)=mT*abb84(10)*abb84(9)
      abb84(9)=2.0_ki*abb84(12)+abb84(9)
      abb84(9)=abb84(9)*abb84(14)
      abb84(12)=4.0_ki*abb84(13)
      abb84(14)=mT**2
      abb84(12)=-abb84(22)*abb84(14)*abb84(12)
      abb84(24)=-abb84(2)*abb84(11)*abb84(7)
      abb84(11)=abb84(11)*c1
      abb84(11)=abb84(24)+abb84(11)
      abb84(13)=2.0_ki*abb84(13)
      abb84(24)=abb84(13)*mT
      abb84(25)=-abb84(24)*spbl5k2*abb84(11)
      abb84(26)=abb84(24)*spbl5l4*abb84(11)
      abb84(21)=abb84(21)-c1
      abb84(10)=abb84(14)*abb84(21)*abb84(10)*abb84(4)*spbl4k2**2
      abb84(27)=abb84(8)*spak1l5*spbl5l4
      abb84(10)=abb84(10)+abb84(27)
      abb84(10)=abb84(10)*abb84(13)
      abb84(27)=spak1l4*spbl4k2
      abb84(28)=spbl5k2*spak1l5
      abb84(27)=abb84(27)-abb84(28)
      abb84(27)=abb84(16)*abb84(27)
      abb84(29)=-abb84(2)*abb84(27)*abb84(7)
      abb84(30)=spal4l5*abb84(18)*spbl4k2
      abb84(31)=abb84(28)*abb84(5)
      abb84(30)=abb84(30)-abb84(31)
      abb84(7)=abb84(2)*abb84(30)*abb84(7)
      abb84(30)=-c1*abb84(30)
      abb84(7)=abb84(30)+abb84(7)
      abb84(7)=mT*abb84(7)
      abb84(27)=c1*abb84(27)
      abb84(7)=abb84(7)+abb84(27)+abb84(29)
      abb84(7)=mT*abb84(7)
      abb84(27)=-abb84(28)*abb84(8)
      abb84(7)=abb84(7)+abb84(27)
      abb84(7)=abb84(7)*abb84(13)
      abb84(14)=abb84(14)*abb84(13)
      abb84(22)=abb84(22)*abb84(14)
      abb84(14)=-abb84(14)*abb84(8)*abb84(18)*spak1l4
      abb84(18)=-abb84(13)*abb84(23)*abb84(8)
      abb84(23)=abb84(8)*abb84(13)
      abb84(6)=abb84(6)*abb84(3)
      abb84(27)=-abb84(19)*abb84(2)
      abb84(6)=abb84(6)+abb84(27)
      abb84(6)=abb84(2)*abb84(1)*abb84(6)
      abb84(19)=-abb84(19)*abb84(1)
      abb84(6)=abb84(19)+abb84(6)
      abb84(6)=abb84(24)*es12*abb84(6)
      abb84(11)=abb84(24)*spbl4k1*abb84(11)
      abb84(16)=abb84(16)*abb84(21)
      abb84(19)=-mT*abb84(5)*abb84(21)
      abb84(16)=2.0_ki*abb84(16)+abb84(19)
      abb84(16)=mT*abb84(16)
      abb84(8)=abb84(16)-abb84(8)
      abb84(8)=abb84(8)*abb84(13)
      R2d84=0.0_ki
      rat2 = rat2 + R2d84
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='84' value='", &
          & R2d84, "'/>"
      end if
   end subroutine
end module p0_ubaru_httbar_abbrevd84h13
