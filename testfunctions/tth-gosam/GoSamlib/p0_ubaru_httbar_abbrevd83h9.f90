module     p0_ubaru_httbar_abbrevd83h9
   use p0_ubaru_httbar_config, only: ki
   use p0_ubaru_httbar_kinematics, only: epstensor
   use p0_ubaru_httbar_globalsh9
   implicit none
   private
   complex(ki), dimension(35), public :: abb83
   complex(ki), public :: R2d83
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
      abb83(1)=sqrt(mT**2)
      abb83(2)=NC**(-1)
      abb83(3)=spak2l4**(-1)
      abb83(4)=spbl4k2**(-1)
      abb83(5)=spak2l5**(-1)
      abb83(6)=abb83(2)*c1
      abb83(7)=abb83(6)*spbl5k2*abb83(1)
      abb83(8)=c2*spbl5k2
      abb83(9)=abb83(8)*abb83(1)
      abb83(7)=-abb83(7)+2.0_ki*abb83(9)
      abb83(7)=abb83(7)*abb83(2)
      abb83(9)=abb83(9)*NC
      abb83(7)=abb83(7)-abb83(9)
      abb83(9)=-spak1l4*abb83(7)
      abb83(10)=TR**2*gHT*e*i_*gs**4
      abb83(11)=4.0_ki*abb83(10)
      abb83(12)=abb83(9)*abb83(11)
      abb83(13)=abb83(1)*NC
      abb83(14)=spak1l4*abb83(5)
      abb83(15)=-abb83(14)*abb83(13)
      abb83(16)=spbl5k2*abb83(4)
      abb83(17)=abb83(16)*spak1l5
      abb83(18)=abb83(13)*abb83(5)
      abb83(19)=abb83(18)*abb83(17)
      abb83(15)=abb83(15)+abb83(19)
      abb83(15)=c2*abb83(15)
      abb83(19)=abb83(1)*c1
      abb83(20)=-abb83(14)*abb83(19)
      abb83(21)=abb83(1)*abb83(5)
      abb83(22)=abb83(21)*c1
      abb83(23)=abb83(22)*abb83(17)
      abb83(20)=abb83(20)+abb83(23)
      abb83(20)=abb83(2)*abb83(20)
      abb83(17)=-abb83(21)*abb83(17)
      abb83(23)=abb83(1)*abb83(14)
      abb83(17)=abb83(23)+abb83(17)
      abb83(23)=2.0_ki*c2
      abb83(17)=abb83(17)*abb83(23)
      abb83(17)=abb83(17)+abb83(20)
      abb83(17)=abb83(2)*abb83(17)
      abb83(20)=abb83(6)-abb83(23)
      abb83(24)=abb83(16)*abb83(3)
      abb83(25)=abb83(24)*spak1l4
      abb83(26)=abb83(25)-abb83(14)
      abb83(26)=-abb83(2)*abb83(26)*abb83(20)
      abb83(27)=abb83(14)*NC
      abb83(25)=-NC*abb83(25)
      abb83(25)=abb83(27)+abb83(25)
      abb83(25)=c2*abb83(25)
      abb83(25)=abb83(25)+abb83(26)
      abb83(25)=mT*abb83(25)
      abb83(15)=abb83(25)+abb83(15)+abb83(17)
      abb83(15)=mT*abb83(15)
      abb83(17)=abb83(1)**2
      abb83(25)=abb83(17)*abb83(14)
      abb83(26)=spbl5k2*spak1l4
      abb83(25)=abb83(25)-abb83(26)
      abb83(25)=abb83(2)*abb83(25)*abb83(20)
      abb83(27)=abb83(17)*abb83(27)
      abb83(28)=-NC*abb83(26)
      abb83(27)=abb83(27)+abb83(28)
      abb83(27)=c2*abb83(27)
      abb83(15)=abb83(15)+abb83(27)+abb83(25)
      abb83(25)=abb83(11)*mT
      abb83(15)=abb83(15)*abb83(25)
      abb83(27)=-abb83(2)*abb83(14)*abb83(20)
      abb83(28)=c2*NC
      abb83(29)=abb83(28)*abb83(14)
      abb83(27)=abb83(27)-abb83(29)
      abb83(29)=-8.0_ki*mT**3*abb83(10)*abb83(27)*abb83(4)*abb83(3)
      abb83(6)=abb83(6)*abb83(26)
      abb83(26)=abb83(8)*spak1l4
      abb83(30)=2.0_ki*abb83(26)-abb83(6)
      abb83(30)=abb83(2)*abb83(30)
      abb83(26)=-NC*abb83(26)
      abb83(26)=abb83(26)+abb83(30)
      abb83(26)=mT*abb83(26)
      abb83(9)=2.0_ki*abb83(9)+abb83(26)
      abb83(9)=abb83(9)*abb83(11)
      abb83(11)=-abb83(27)*abb83(25)
      abb83(26)=abb83(28)*abb83(16)
      abb83(30)=-abb83(2)*abb83(16)*abb83(20)
      abb83(26)=abb83(26)-abb83(30)
      abb83(30)=-abb83(26)*spak1l4*spbl5l4
      abb83(20)=abb83(20)*abb83(2)
      abb83(20)=abb83(28)+abb83(20)
      abb83(28)=mT**2
      abb83(31)=-abb83(28)*abb83(20)*abb83(3)*spak1k2*abb83(16)**2
      abb83(30)=abb83(31)+abb83(30)
      abb83(31)=2.0_ki*mT
      abb83(31)=abb83(31)*abb83(10)
      abb83(30)=abb83(30)*abb83(31)
      abb83(10)=2.0_ki*abb83(10)
      abb83(32)=abb83(10)*spak1l5*abb83(7)
      abb83(33)=spak1k2*abb83(21)
      abb83(24)=abb83(24)*spak1k2
      abb83(34)=-abb83(1)*abb83(24)
      abb83(33)=abb83(33)+abb83(34)
      abb83(33)=abb83(33)*abb83(23)
      abb83(34)=abb83(24)*abb83(19)
      abb83(35)=-spak1k2*abb83(22)
      abb83(34)=abb83(35)+abb83(34)
      abb83(34)=abb83(2)*abb83(34)
      abb83(33)=abb83(33)+abb83(34)
      abb83(33)=abb83(2)*abb83(33)
      abb83(24)=abb83(13)*abb83(24)
      abb83(34)=-spak1k2*abb83(18)
      abb83(24)=abb83(34)+abb83(24)
      abb83(24)=c2*abb83(24)
      abb83(24)=abb83(24)+abb83(33)
      abb83(24)=abb83(24)*abb83(28)
      abb83(28)=-spak1k2*abb83(7)
      abb83(24)=abb83(24)+abb83(28)
      abb83(24)=abb83(24)*abb83(10)
      abb83(6)=abb83(6)*abb83(5)
      abb83(8)=abb83(14)*abb83(8)
      abb83(6)=-abb83(6)+2.0_ki*abb83(8)
      abb83(14)=abb83(2)*spal4l5*abb83(6)
      abb83(8)=abb83(8)*NC
      abb83(28)=-spal4l5*abb83(8)
      abb83(14)=abb83(28)+abb83(14)
      abb83(14)=abb83(14)*abb83(31)
      abb83(27)=abb83(27)*abb83(31)
      abb83(6)=-abb83(2)*spak1l5*abb83(6)
      abb83(8)=spak1l5*abb83(8)
      abb83(6)=abb83(8)+abb83(6)
      abb83(6)=abb83(6)*abb83(31)
      abb83(8)=-abb83(10)*spal4l5*abb83(7)
      abb83(28)=-abb83(26)*abb83(31)
      abb83(17)=-abb83(17)*abb83(26)
      abb83(31)=spak2l4*abb83(21)
      abb83(33)=-abb83(1)*abb83(16)
      abb83(31)=abb83(31)+abb83(33)
      abb83(23)=abb83(31)*abb83(23)
      abb83(19)=abb83(16)*abb83(19)
      abb83(22)=-spak2l4*abb83(22)
      abb83(19)=abb83(22)+abb83(19)
      abb83(19)=abb83(2)*abb83(19)
      abb83(19)=abb83(23)+abb83(19)
      abb83(19)=abb83(2)*abb83(19)
      abb83(13)=abb83(16)*abb83(13)
      abb83(16)=-spak2l4*abb83(18)
      abb83(13)=abb83(16)+abb83(13)
      abb83(13)=c2*abb83(13)
      abb83(13)=abb83(13)+abb83(19)
      abb83(13)=mT*abb83(13)
      abb83(13)=abb83(13)+abb83(17)
      abb83(13)=mT*abb83(13)
      abb83(7)=-spak2l4*abb83(7)
      abb83(7)=abb83(13)+abb83(7)
      abb83(7)=abb83(7)*abb83(10)
      abb83(10)=-abb83(20)*abb83(21)*abb83(4)
      abb83(13)=mT*abb83(20)*abb83(5)*abb83(4)
      abb83(10)=2.0_ki*abb83(10)+abb83(13)
      abb83(10)=mT*abb83(10)
      abb83(10)=abb83(10)-abb83(26)
      abb83(10)=abb83(10)*abb83(25)
      R2d83=0.0_ki
      rat2 = rat2 + R2d83
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='83' value='", &
          & R2d83, "'/>"
      end if
   end subroutine
end module p0_ubaru_httbar_abbrevd83h9
