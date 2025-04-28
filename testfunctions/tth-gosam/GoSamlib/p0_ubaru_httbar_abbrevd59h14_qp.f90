module     p0_ubaru_httbar_abbrevd59h14_qp
   use p0_ubaru_httbar_config, only: ki => ki_qp
   use p0_ubaru_httbar_kinematics_qp, only: epstensor
   use p0_ubaru_httbar_globalsh14_qp
   implicit none
   private
   complex(ki), dimension(26), public :: abb59
   complex(ki), public :: R2d59
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
      abb59(1)=sqrt(mT**2)
      abb59(2)=NC**(-1)
      abb59(3)=es12**(-1)
      abb59(4)=spak2l5**(-1)
      abb59(5)=spak2l4**(-1)
      abb59(6)=spbl4k2**(-1)
      abb59(7)=spbl5k2**(-1)
      abb59(8)=c2*abb59(2)
      abb59(9)=c1*abb59(2)**2
      abb59(8)=abb59(8)-abb59(9)
      abb59(8)=abb59(8)*i_*e*gHT*abb59(3)*TR**2*gs**4
      abb59(9)=-spak2l3*abb59(8)
      abb59(10)=abb59(1)**2
      abb59(11)=abb59(9)*abb59(10)
      abb59(12)=-abb59(1)*abb59(9)
      abb59(13)=abb59(12)*mT
      abb59(14)=abb59(13)-abb59(11)
      abb59(14)=spbl4l3*abb59(14)
      abb59(10)=abb59(10)*abb59(8)
      abb59(15)=abb59(10)*spbl5l4
      abb59(16)=spak2l5*abb59(15)
      abb59(14)=abb59(16)+abb59(14)
      abb59(14)=spbl5k1*abb59(14)
      abb59(16)=abb59(4)*mT
      abb59(17)=spbl4l3*abb59(12)*abb59(16)
      abb59(15)=abb59(17)+abb59(15)
      abb59(18)=spak2l4*spbl4k1
      abb59(15)=abb59(15)*abb59(18)
      abb59(11)=spbl5l3*spbl4k1*abb59(11)
      abb59(13)=abb59(13)*spbl3k1
      abb59(19)=spbl5l4*abb59(13)
      abb59(11)=abb59(19)+abb59(15)+abb59(11)+abb59(14)
      abb59(11)=2.0_ki*abb59(11)
      abb59(14)=abb59(9)*spbl4l3
      abb59(15)=abb59(14)*spbl5k1
      abb59(19)=-4.0_ki*abb59(15)
      abb59(9)=abb59(9)*spbl5l3
      abb59(20)=abb59(9)*spbl4k1
      abb59(21)=4.0_ki*abb59(20)
      abb59(22)=-spak2l5*spbl5k1
      abb59(18)=abb59(22)-abb59(18)
      abb59(18)=-abb59(18)*abb59(8)*spbl5l4
      abb59(15)=-abb59(15)+abb59(20)+abb59(18)
      abb59(15)=2.0_ki*abb59(15)
      abb59(18)=2.0_ki*spbl5k1
      abb59(20)=abb59(9)*abb59(18)
      abb59(22)=2.0_ki*spbl4k1
      abb59(23)=-abb59(14)*abb59(22)
      abb59(24)=abb59(6)*abb59(5)
      abb59(25)=abb59(7)*abb59(4)
      abb59(24)=abb59(24)+abb59(25)
      abb59(24)=abb59(24)*spbk2k1*mT**2
      abb59(14)=-abb59(14)*abb59(24)
      abb59(25)=abb59(10)*spbl4k1
      abb59(13)=-abb59(5)*abb59(13)
      abb59(13)=abb59(13)-3.0_ki*abb59(25)+abb59(14)
      abb59(13)=2.0_ki*abb59(13)
      abb59(14)=-abb59(8)*abb59(22)
      abb59(9)=abb59(9)*abb59(24)
      abb59(24)=abb59(1)*abb59(8)
      abb59(25)=2.0_ki*mT
      abb59(26)=abb59(24)*abb59(25)
      abb59(10)=3.0_ki*abb59(10)+abb59(26)
      abb59(10)=spbl5k1*abb59(10)
      abb59(16)=abb59(24)*abb59(16)
      abb59(22)=spak2l4*abb59(22)*abb59(16)
      abb59(9)=abb59(10)+abb59(22)+abb59(9)
      abb59(9)=2.0_ki*abb59(9)
      abb59(8)=abb59(8)*abb59(18)
      abb59(10)=-abb59(5)*spbl5k1*abb59(12)*abb59(25)
      abb59(12)=-spbl5l4*abb59(26)
      abb59(12)=abb59(17)+abb59(12)
      abb59(12)=2.0_ki*abb59(12)
      abb59(17)=4.0_ki*abb59(5)*mT*abb59(24)
      abb59(16)=4.0_ki*abb59(16)
      R2d59=0.0_ki
      rat2 = rat2 + R2d59
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='59' value='", &
          & R2d59, "'/>"
      end if
   end subroutine
end module p0_ubaru_httbar_abbrevd59h14_qp
