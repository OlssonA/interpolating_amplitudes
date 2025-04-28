module     p0_ubaru_httbar_abbrevd39h14
   use p0_ubaru_httbar_config, only: ki
   use p0_ubaru_httbar_kinematics, only: epstensor
   use p0_ubaru_httbar_globalsh14
   implicit none
   private
   complex(ki), dimension(27), public :: abb39
   complex(ki), public :: R2d39
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
      abb39(1)=1.0_ki/(mH**2+mT**2-es34-es45+es12)
      abb39(2)=NC**(-1)
      abb39(3)=es12**(-1)
      abb39(4)=1.0_ki/(-mT**2+es34)
      abb39(5)=spak2l4**(-1)
      abb39(6)=spak2l3**(-1)
      abb39(7)=spbl3k2**(-1)
      abb39(8)=spak2l5**(-1)
      abb39(9)=sqrt(mT**2)
      abb39(10)=Nfrat*abb39(2)
      abb39(11)=10.0_ki*abb39(10)
      abb39(12)=abb39(11)+1.0_ki
      abb39(12)=abb39(12)*c1
      abb39(13)=10.0_ki*Nfrat
      abb39(14)=abb39(13)+NC
      abb39(14)=abb39(14)*c2
      abb39(12)=abb39(12)-abb39(14)
      abb39(14)=spak2l3*spbl4l3
      abb39(15)=abb39(14)*spbl5k1
      abb39(16)=-abb39(4)*abb39(15)*abb39(12)
      abb39(17)=spak2l3*spbl5l3
      abb39(18)=abb39(17)*spbl4k1
      abb39(12)=-abb39(1)*abb39(18)*abb39(12)
      abb39(12)=abb39(16)+abb39(12)
      abb39(16)=i_*gs**4*gHT*e
      abb39(19)=abb39(3)*TR**2*abb39(16)
      abb39(20)=1.0_ki/3.0_ki*abb39(12)*abb39(19)
      abb39(15)=-abb39(4)*abb39(15)
      abb39(18)=-abb39(1)*abb39(18)
      abb39(15)=abb39(15)+abb39(18)
      abb39(18)=c2*NC
      abb39(18)=abb39(18)-c1
      abb39(15)=abb39(18)*abb39(15)
      abb39(18)=-5.0_ki*abb39(15)*abb39(19)
      abb39(19)=abb39(3)*TR
      abb39(16)=abb39(19)**2*abb39(16)
      abb39(19)=2.0_ki*abb39(16)
      abb39(15)=abb39(15)*abb39(19)
      abb39(12)=abb39(12)*abb39(19)
      abb39(19)=spbl5k1*spbl4l3
      abb39(21)=-abb39(4)*abb39(19)
      abb39(22)=spbl4k1*spbl5l3
      abb39(23)=-abb39(1)*abb39(22)
      abb39(21)=abb39(21)+abb39(23)
      abb39(23)=c2*Nfrat
      abb39(10)=abb39(10)*c1
      abb39(10)=abb39(23)-abb39(10)
      abb39(23)=10.0_ki*abb39(16)
      abb39(21)=-abb39(23)*abb39(21)*spak1k2*abb39(10)
      abb39(10)=-abb39(23)*spbk2k1*abb39(10)
      abb39(23)=abb39(4)*abb39(14)*abb39(10)
      abb39(10)=abb39(1)*abb39(17)*abb39(10)
      abb39(14)=abb39(14)*spbl5k2
      abb39(19)=abb39(19)*spak1l3
      abb39(24)=abb39(7)*abb39(6)*mH**2
      abb39(25)=abb39(24)*spak1k2
      abb39(26)=spbl4k2*abb39(25)*spbl5k1
      abb39(14)=abb39(26)+abb39(14)+abb39(19)
      abb39(19)=-abb39(11)*abb39(26)
      abb39(19)=3.0_ki/2.0_ki*abb39(14)+abb39(19)
      abb39(19)=c1*abb39(19)
      abb39(14)=-NC*abb39(14)
      abb39(26)=abb39(13)*abb39(26)
      abb39(14)=3.0_ki/2.0_ki*abb39(14)+abb39(26)
      abb39(14)=c2*abb39(14)
      abb39(14)=abb39(19)+abb39(14)
      abb39(14)=abb39(4)*abb39(14)
      abb39(17)=abb39(17)*spbl4k2
      abb39(19)=abb39(22)*spak1l3
      abb39(22)=spbl5k2*abb39(25)*spbl4k1
      abb39(17)=abb39(22)+abb39(17)+abb39(19)
      abb39(11)=-abb39(11)*abb39(22)
      abb39(11)=3.0_ki/2.0_ki*abb39(17)+abb39(11)
      abb39(11)=c1*abb39(11)
      abb39(17)=-NC*abb39(17)
      abb39(19)=abb39(13)*abb39(22)
      abb39(17)=3.0_ki/2.0_ki*abb39(17)+abb39(19)
      abb39(17)=c2*abb39(17)
      abb39(11)=abb39(11)+abb39(17)
      abb39(11)=abb39(1)*abb39(11)
      abb39(17)=abb39(8)*spbl4k1
      abb39(19)=abb39(5)*spbl5k1
      abb39(17)=abb39(17)+abb39(19)
      abb39(19)=abb39(9)*abb39(17)
      abb39(22)=abb39(19)*abb39(13)*spak1k2
      abb39(25)=3.0_ki/2.0_ki*spak1k2
      abb39(19)=abb39(19)*abb39(25)
      abb39(25)=NC*abb39(19)
      abb39(25)=abb39(25)-abb39(22)
      abb39(25)=abb39(25)*c2
      abb39(22)=abb39(2)*abb39(22)
      abb39(19)=abb39(22)-abb39(19)
      abb39(19)=abb39(19)*c1
      abb39(19)=abb39(25)+abb39(19)
      abb39(22)=abb39(1)+abb39(4)
      abb39(19)=-abb39(19)*abb39(22)
      abb39(25)=abb39(8)*spak2l3*abb39(5)
      abb39(26)=spbl3k1*abb39(25)
      abb39(17)=abb39(17)+abb39(26)
      abb39(17)=spak1k2*abb39(17)
      abb39(26)=abb39(13)*abb39(2)*abb39(17)
      abb39(26)=abb39(26)-3.0_ki/2.0_ki*abb39(17)
      abb39(26)=abb39(26)*c1
      abb39(13)=-abb39(13)+3.0_ki/2.0_ki*NC
      abb39(13)=c2*abb39(17)*abb39(13)
      abb39(13)=abb39(13)+abb39(26)
      abb39(17)=abb39(22)*mT
      abb39(13)=-abb39(13)*abb39(17)
      abb39(13)=abb39(13)+abb39(19)
      abb39(13)=mT*abb39(13)
      abb39(11)=abb39(13)+abb39(14)+abb39(11)
      abb39(11)=abb39(11)*abb39(16)
      abb39(13)=5.0_ki*Nfrat
      abb39(14)=abb39(13)*abb39(2)
      abb39(14)=abb39(14)-1.0_ki
      abb39(14)=abb39(14)*c1
      abb39(13)=abb39(13)-NC
      abb39(13)=abb39(13)*c2
      abb39(13)=abb39(14)-abb39(13)
      abb39(14)=4.0_ki*abb39(16)
      abb39(16)=abb39(4)*abb39(14)*spbl4l3*abb39(13)
      abb39(19)=abb39(1)*abb39(14)*spbl5l3*abb39(13)
      abb39(26)=abb39(22)*abb39(13)*abb39(5)*abb39(9)
      abb39(27)=abb39(17)*abb39(5)*abb39(13)
      abb39(26)=abb39(27)+abb39(26)
      abb39(26)=mT*abb39(26)
      abb39(27)=abb39(4)*abb39(13)*abb39(24)*spbl4k2
      abb39(26)=abb39(27)+abb39(26)
      abb39(26)=abb39(26)*abb39(14)
      abb39(27)=abb39(22)*abb39(13)*abb39(8)*abb39(9)
      abb39(17)=abb39(17)*abb39(8)*abb39(13)
      abb39(17)=abb39(17)+abb39(27)
      abb39(17)=mT*abb39(17)
      abb39(24)=abb39(1)*abb39(13)*abb39(24)*spbl5k2
      abb39(17)=abb39(24)+abb39(17)
      abb39(17)=abb39(17)*abb39(14)
      abb39(13)=mT**2*abb39(14)*abb39(22)*abb39(25)*abb39(13)
      R2d39=abb39(20)
      rat2 = rat2 + R2d39
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='39' value='", &
          & R2d39, "'/>"
      end if
   end subroutine
end module p0_ubaru_httbar_abbrevd39h14
