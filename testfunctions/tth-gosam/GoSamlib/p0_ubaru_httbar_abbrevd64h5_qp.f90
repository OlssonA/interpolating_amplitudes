module     p0_ubaru_httbar_abbrevd64h5_qp
   use p0_ubaru_httbar_config, only: ki => ki_qp
   use p0_ubaru_httbar_kinematics_qp, only: epstensor
   use p0_ubaru_httbar_globalsh5_qp
   implicit none
   private
   complex(ki), dimension(35), public :: abb64
   complex(ki), public :: R2d64
   public :: init_abbrev
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
contains
   subroutine     init_abbrev()
      use p0_ubaru_httbar_config, only: deltaOS, &
     &    logfile, debug_nlo_diagrams
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_model_qp
      use p0_ubaru_httbar_color_qp, only: TR
      use p0_ubaru_httbar_globalsl1_qp, only: epspow
      implicit none
      abb64(1)=1.0_ki/(-mT**2+es34)
      abb64(2)=NC**(-1)
      abb64(3)=spak2l3**(-1)
      abb64(4)=spbl3k2**(-1)
      abb64(5)=spak2l4**(-1)
      abb64(6)=spak2l5**(-1)
      abb64(7)=spbl5k2**(-1)
      abb64(8)=sqrt(mT**2)
      abb64(9)=i_*e*gHT*abb64(1)*TR**2*gs**4
      abb64(10)=abb64(9)*abb64(2)
      abb64(11)=spak1l5*abb64(10)
      abb64(12)=mT**3
      abb64(13)=abb64(11)*abb64(12)
      abb64(11)=abb64(11)*abb64(8)
      abb64(14)=mT**2
      abb64(15)=abb64(14)*abb64(11)
      abb64(15)=abb64(13)+abb64(15)
      abb64(16)=2.0_ki*c2
      abb64(15)=abb64(15)*abb64(16)
      abb64(17)=abb64(2)**2
      abb64(17)=abb64(17)+1.0_ki
      abb64(18)=abb64(9)*abb64(17)
      abb64(19)=spak1l5*abb64(18)
      abb64(20)=abb64(12)*abb64(19)
      abb64(21)=abb64(20)*c1
      abb64(22)=abb64(8)*c1
      abb64(19)=abb64(19)*abb64(22)
      abb64(23)=-abb64(14)*abb64(19)
      abb64(15)=abb64(15)-abb64(21)+abb64(23)
      abb64(23)=spbl4k2*abb64(7)
      abb64(15)=abb64(15)*abb64(23)
      abb64(24)=abb64(5)*c1
      abb64(20)=-abb64(20)*abb64(24)
      abb64(25)=abb64(16)*abb64(5)
      abb64(26)=abb64(25)*abb64(13)
      abb64(20)=abb64(20)+abb64(26)
      abb64(26)=spbl3k2*spak2l3
      abb64(27)=abb64(26)*abb64(7)
      abb64(20)=abb64(20)*abb64(27)
      abb64(15)=abb64(20)+abb64(15)
      abb64(15)=abb64(6)*abb64(15)
      abb64(9)=mT*abb64(9)*spak1l5
      abb64(20)=abb64(9)*abb64(2)
      abb64(28)=abb64(11)-abb64(20)
      abb64(28)=abb64(28)*abb64(16)
      abb64(9)=abb64(9)*abb64(17)
      abb64(17)=abb64(9)*c1
      abb64(28)=abb64(28)-abb64(19)+abb64(17)
      abb64(29)=abb64(3)*abb64(4)*mH**2
      abb64(30)=spbl4k2*abb64(28)*abb64(29)
      abb64(31)=abb64(8)**2
      abb64(32)=-abb64(31)*abb64(17)
      abb64(21)=abb64(21)+abb64(32)
      abb64(21)=abb64(5)*abb64(21)
      abb64(31)=abb64(31)*abb64(20)
      abb64(13)=-abb64(13)+abb64(31)
      abb64(13)=abb64(13)*abb64(25)
      abb64(13)=abb64(30)+abb64(21)+abb64(13)+abb64(15)
      abb64(13)=4.0_ki*abb64(13)
      abb64(11)=abb64(11)+abb64(20)
      abb64(11)=abb64(11)*abb64(16)
      abb64(11)=-abb64(17)+abb64(11)-abb64(19)
      abb64(15)=spbl4k2*abb64(11)
      abb64(9)=abb64(9)*abb64(24)
      abb64(17)=abb64(25)*abb64(20)
      abb64(9)=abb64(9)-abb64(17)
      abb64(17)=-abb64(9)*abb64(26)
      abb64(15)=abb64(15)+abb64(17)
      abb64(15)=4.0_ki*abb64(15)
      abb64(17)=2.0_ki*abb64(11)
      abb64(19)=spbl5k2*abb64(17)
      abb64(9)=abb64(9)*spak2l3
      abb64(20)=2.0_ki*abb64(9)
      abb64(21)=-spbl5k2*abb64(20)
      abb64(24)=-spbl5l4*abb64(11)
      abb64(26)=spbl5l3*abb64(9)
      abb64(24)=abb64(24)+abb64(26)
      abb64(24)=2.0_ki*abb64(24)
      abb64(26)=2.0_ki*spbl4l3
      abb64(28)=abb64(28)*abb64(26)
      abb64(17)=spbk2k1*abb64(17)
      abb64(20)=-spbk2k1*abb64(20)
      abb64(30)=abb64(14)*abb64(18)
      abb64(31)=c1*abb64(30)
      abb64(32)=mT*abb64(18)
      abb64(33)=abb64(32)*abb64(22)
      abb64(14)=abb64(14)*abb64(10)
      abb64(34)=abb64(10)*mT
      abb64(35)=-abb64(8)*abb64(34)
      abb64(35)=-abb64(14)+abb64(35)
      abb64(35)=abb64(35)*abb64(16)
      abb64(31)=abb64(35)+abb64(31)+abb64(33)
      abb64(31)=abb64(23)*abb64(8)*abb64(31)
      abb64(11)=-spbl4k1*abb64(11)
      abb64(22)=abb64(30)*abb64(22)
      abb64(30)=abb64(5)*abb64(22)
      abb64(14)=abb64(14)*abb64(8)
      abb64(33)=-abb64(25)*abb64(14)
      abb64(30)=abb64(30)+abb64(33)
      abb64(27)=abb64(30)*abb64(27)
      abb64(9)=spbl3k1*abb64(9)
      abb64(9)=abb64(9)+abb64(11)+abb64(31)+abb64(27)
      abb64(9)=2.0_ki*abb64(9)
      abb64(11)=c1*abb64(12)*abb64(18)
      abb64(11)=abb64(11)+abb64(22)
      abb64(11)=abb64(5)*abb64(11)
      abb64(10)=-abb64(12)*abb64(10)
      abb64(10)=abb64(10)-abb64(14)
      abb64(10)=abb64(10)*abb64(25)
      abb64(10)=abb64(11)+abb64(10)
      abb64(10)=abb64(7)*abb64(10)
      abb64(11)=abb64(34)*abb64(16)
      abb64(12)=abb64(32)*c1
      abb64(11)=abb64(11)-abb64(12)
      abb64(12)=-abb64(11)*abb64(23)*abb64(29)
      abb64(10)=abb64(10)+abb64(12)
      abb64(10)=4.0_ki*abb64(10)
      abb64(11)=-abb64(7)*abb64(11)*abb64(26)
      R2d64=0.0_ki
      rat2 = rat2 + R2d64
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='64' value='", &
          & R2d64, "'/>"
      end if
   end subroutine
end module p0_ubaru_httbar_abbrevd64h5_qp
