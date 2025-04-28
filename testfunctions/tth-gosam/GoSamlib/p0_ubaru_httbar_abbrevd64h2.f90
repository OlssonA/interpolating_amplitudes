module     p0_ubaru_httbar_abbrevd64h2
   use p0_ubaru_httbar_config, only: ki
   use p0_ubaru_httbar_kinematics, only: epstensor
   use p0_ubaru_httbar_globalsh2
   implicit none
   private
   complex(ki), dimension(34), public :: abb64
   complex(ki), public :: R2d64
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
      abb64(1)=1.0_ki/(-mT**2+es34)
      abb64(2)=NC**(-1)
      abb64(3)=spak2l5**(-1)
      abb64(4)=spbl5k2**(-1)
      abb64(5)=spbl4k2**(-1)
      abb64(6)=sqrt(mT**2)
      abb64(7)=spak2l3**(-1)
      abb64(8)=spbl3k2**(-1)
      abb64(9)=abb64(2)**2
      abb64(9)=abb64(9)+1.0_ki
      abb64(10)=i_*e*gHT*abb64(1)*TR**2*gs**4
      abb64(11)=abb64(10)*abb64(9)
      abb64(12)=mT**3
      abb64(13)=abb64(11)*abb64(12)
      abb64(14)=spbk2k1*abb64(4)
      abb64(15)=abb64(14)*c1
      abb64(16)=abb64(13)*abb64(15)
      abb64(17)=abb64(10)*mT**2
      abb64(9)=abb64(17)*abb64(9)
      abb64(18)=abb64(15)*abb64(6)
      abb64(19)=abb64(9)*abb64(18)
      abb64(16)=abb64(16)+abb64(19)
      abb64(16)=abb64(5)*abb64(16)
      abb64(19)=abb64(10)*abb64(2)
      abb64(20)=-abb64(12)*abb64(14)*abb64(19)
      abb64(17)=abb64(17)*abb64(2)
      abb64(21)=abb64(17)*abb64(14)
      abb64(22)=-abb64(6)*abb64(21)
      abb64(20)=abb64(20)+abb64(22)
      abb64(22)=2.0_ki*c2
      abb64(23)=abb64(22)*abb64(5)
      abb64(20)=abb64(20)*abb64(23)
      abb64(16)=abb64(16)+abb64(20)
      abb64(16)=abb64(6)*abb64(16)
      abb64(20)=abb64(11)*c1
      abb64(24)=mT**4
      abb64(25)=abb64(24)*abb64(20)
      abb64(26)=abb64(6)*c1
      abb64(13)=abb64(13)*abb64(26)
      abb64(13)=abb64(25)+abb64(13)
      abb64(27)=spbk2k1*abb64(4)**2
      abb64(13)=abb64(27)*abb64(13)
      abb64(28)=abb64(27)*abb64(19)
      abb64(24)=abb64(28)*abb64(24)
      abb64(12)=-abb64(6)*abb64(12)*abb64(28)
      abb64(12)=-abb64(24)+abb64(12)
      abb64(12)=abb64(12)*abb64(22)
      abb64(12)=abb64(12)+abb64(13)
      abb64(12)=spak2l4*abb64(12)
      abb64(13)=abb64(5)*abb64(27)*abb64(25)
      abb64(24)=-abb64(23)*abb64(24)
      abb64(13)=abb64(13)+abb64(24)
      abb64(24)=spak2l3*spbl3k2
      abb64(13)=abb64(13)*abb64(24)
      abb64(12)=abb64(13)+abb64(12)
      abb64(12)=abb64(3)*abb64(12)
      abb64(13)=abb64(2)*mT
      abb64(25)=abb64(10)*abb64(13)*abb64(6)
      abb64(14)=abb64(14)*abb64(25)
      abb64(27)=abb64(14)*abb64(22)
      abb64(28)=mT*abb64(11)
      abb64(18)=abb64(28)*abb64(18)
      abb64(27)=abb64(27)-abb64(18)
      abb64(29)=spak2l4*mH**2*abb64(8)*abb64(7)
      abb64(30)=-abb64(27)*abb64(29)
      abb64(12)=abb64(30)+abb64(12)+abb64(16)
      abb64(12)=4.0_ki*abb64(12)
      abb64(14)=abb64(14)+abb64(21)
      abb64(14)=abb64(14)*abb64(22)
      abb64(15)=abb64(9)*abb64(15)
      abb64(14)=-abb64(14)+abb64(18)+abb64(15)
      abb64(16)=spak2l4*abb64(14)
      abb64(15)=abb64(15)*abb64(5)
      abb64(18)=abb64(21)*abb64(23)
      abb64(15)=abb64(15)-abb64(18)
      abb64(18)=abb64(15)*abb64(24)
      abb64(16)=abb64(16)+abb64(18)
      abb64(16)=4.0_ki*abb64(16)
      abb64(18)=abb64(19)*abb64(22)
      abb64(18)=abb64(18)-abb64(20)
      abb64(20)=2.0_ki*spal3l4
      abb64(21)=abb64(20)*abb64(18)
      abb64(30)=spbl5k1*spak2l5
      abb64(31)=-abb64(30)*abb64(21)
      abb64(18)=abb64(29)*abb64(18)
      abb64(9)=abb64(9)*c1
      abb64(29)=abb64(26)*abb64(28)
      abb64(32)=abb64(9)+abb64(29)
      abb64(33)=abb64(32)*abb64(5)
      abb64(10)=abb64(10)*abb64(13)
      abb64(13)=abb64(6)*abb64(10)
      abb64(13)=abb64(13)+abb64(17)
      abb64(34)=abb64(13)*abb64(23)
      abb64(18)=abb64(18)+abb64(34)-abb64(33)
      abb64(18)=2.0_ki*abb64(18)
      abb64(30)=-abb64(30)*abb64(18)
      abb64(28)=c1*abb64(28)
      abb64(11)=abb64(11)*abb64(26)
      abb64(19)=-abb64(6)*abb64(19)
      abb64(10)=-abb64(10)+abb64(19)
      abb64(10)=abb64(10)*abb64(22)
      abb64(10)=abb64(10)+abb64(28)+abb64(11)
      abb64(10)=spak2l4*abb64(6)*abb64(10)
      abb64(11)=-abb64(22)*abb64(25)
      abb64(11)=abb64(29)+abb64(11)
      abb64(11)=abb64(24)*abb64(5)*abb64(11)
      abb64(10)=abb64(10)+abb64(11)
      abb64(10)=2.0_ki*abb64(10)
      abb64(11)=2.0_ki*spak2l5
      abb64(19)=abb64(14)*abb64(11)
      abb64(13)=-abb64(22)*abb64(13)
      abb64(13)=abb64(13)+abb64(32)
      abb64(22)=2.0_ki*es12
      abb64(13)=abb64(22)*abb64(4)*abb64(13)
      abb64(15)=abb64(15)*spbl3k2
      abb64(11)=abb64(11)*abb64(15)
      abb64(17)=abb64(23)*abb64(17)
      abb64(9)=abb64(5)*abb64(9)
      abb64(9)=abb64(9)-abb64(17)
      abb64(9)=spbl3k2*abb64(22)*abb64(4)*abb64(9)
      abb64(17)=-spal4l5*abb64(14)
      abb64(22)=-spal3l5*abb64(15)
      abb64(17)=abb64(17)+abb64(22)
      abb64(17)=2.0_ki*abb64(17)
      abb64(20)=-abb64(27)*abb64(20)
      abb64(14)=-spak1l4*abb64(14)
      abb64(15)=-spak1l3*abb64(15)
      abb64(14)=abb64(14)+abb64(15)
      abb64(14)=2.0_ki*abb64(14)
      R2d64=0.0_ki
      rat2 = rat2 + R2d64
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='64' value='", &
          & R2d64, "'/>"
      end if
   end subroutine
end module p0_ubaru_httbar_abbrevd64h2
