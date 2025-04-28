module     p0_ubaru_httbar_abbrevd83h2_qp
   use p0_ubaru_httbar_config, only: ki => ki_qp
   use p0_ubaru_httbar_kinematics_qp, only: epstensor
   use p0_ubaru_httbar_globalsh2_qp
   implicit none
   private
   complex(ki), dimension(29), public :: abb83
   complex(ki), public :: R2d83
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
      abb83(1)=sqrt(mT**2)
      abb83(2)=NC**(-1)
      abb83(3)=spbl4k2**(-1)
      abb83(4)=spak2l4**(-1)
      abb83(5)=spbl5k2**(-1)
      abb83(6)=abb83(2)*c1
      abb83(7)=abb83(6)-2.0_ki*c2
      abb83(8)=-abb83(2)*abb83(7)*spak2l5
      abb83(9)=c2*NC
      abb83(10)=spak2l5*abb83(9)
      abb83(8)=abb83(8)-abb83(10)
      abb83(11)=abb83(3)*spbk2k1
      abb83(12)=abb83(11)*abb83(1)
      abb83(13)=-abb83(12)*abb83(8)
      abb83(14)=TR**2*gHT*e*i_*gs**4
      abb83(15)=abb83(14)*mT
      abb83(15)=4.0_ki*abb83(15)
      abb83(16)=abb83(13)*abb83(15)
      abb83(17)=abb83(6)*abb83(3)
      abb83(18)=c2*abb83(3)
      abb83(19)=abb83(17)-2.0_ki*abb83(18)
      abb83(20)=abb83(1)*abb83(5)
      abb83(21)=spbk2k1*abb83(20)
      abb83(22)=abb83(4)*abb83(12)*spak2l5
      abb83(21)=abb83(21)-abb83(22)
      abb83(19)=abb83(2)*abb83(21)*abb83(19)
      abb83(21)=abb83(18)*NC*abb83(21)
      abb83(22)=spbk2k1*abb83(3)**2
      abb83(6)=abb83(22)*abb83(6)*abb83(4)
      abb83(23)=spak2l5*abb83(6)
      abb83(22)=abb83(4)*c2*abb83(22)
      abb83(24)=abb83(22)*spak2l5
      abb83(23)=-2.0_ki*abb83(24)+abb83(23)
      abb83(23)=abb83(2)*abb83(23)
      abb83(24)=NC*abb83(24)
      abb83(23)=abb83(24)+abb83(23)
      abb83(23)=mT*abb83(23)
      abb83(19)=abb83(23)+abb83(21)+abb83(19)
      abb83(19)=mT*abb83(19)
      abb83(21)=abb83(11)*abb83(5)
      abb83(23)=-abb83(2)*abb83(21)*abb83(7)
      abb83(24)=abb83(21)*abb83(9)
      abb83(23)=abb83(23)-abb83(24)
      abb83(24)=abb83(1)**2
      abb83(25)=abb83(24)*abb83(23)
      abb83(19)=abb83(19)+abb83(25)
      abb83(19)=mT*abb83(19)
      abb83(19)=abb83(19)-abb83(13)
      abb83(19)=abb83(19)*abb83(15)
      abb83(6)=-abb83(5)*abb83(6)
      abb83(22)=abb83(22)*abb83(5)
      abb83(6)=2.0_ki*abb83(22)+abb83(6)
      abb83(6)=abb83(2)*abb83(6)
      abb83(22)=-NC*abb83(22)
      abb83(6)=abb83(22)+abb83(6)
      abb83(6)=8.0_ki*abb83(6)*mT**4*abb83(14)
      abb83(22)=abb83(7)*abb83(2)
      abb83(25)=spak2l5*abb83(22)
      abb83(10)=abb83(10)+abb83(25)
      abb83(10)=mT*abb83(11)*abb83(10)
      abb83(10)=-2.0_ki*abb83(13)+abb83(10)
      abb83(10)=abb83(10)*abb83(15)
      abb83(13)=4.0_ki*abb83(14)
      abb83(15)=mT**2
      abb83(13)=abb83(23)*abb83(15)*abb83(13)
      abb83(25)=-abb83(2)*abb83(12)*abb83(7)
      abb83(12)=abb83(12)*abb83(9)
      abb83(12)=abb83(25)-abb83(12)
      abb83(14)=2.0_ki*abb83(14)
      abb83(25)=abb83(14)*mT
      abb83(26)=abb83(25)*spak2l4*abb83(12)
      abb83(18)=abb83(18)*abb83(1)
      abb83(17)=-abb83(1)*abb83(17)
      abb83(17)=2.0_ki*abb83(18)+abb83(17)
      abb83(17)=abb83(2)*abb83(17)
      abb83(18)=-NC*abb83(18)
      abb83(17)=abb83(18)+abb83(17)
      abb83(17)=abb83(25)*es12*abb83(17)
      abb83(18)=abb83(9)+abb83(22)
      abb83(11)=-abb83(15)*abb83(11)*abb83(18)*abb83(4)*spak2l5**2
      abb83(22)=abb83(8)*spbl4k1*spal4l5
      abb83(11)=abb83(11)+abb83(22)
      abb83(11)=abb83(11)*abb83(14)
      abb83(22)=abb83(8)*abb83(14)
      abb83(27)=spbl5l4*abb83(21)*spak2l5
      abb83(28)=spak2l4*spbl4k1
      abb83(29)=abb83(28)*abb83(5)
      abb83(27)=abb83(27)+abb83(29)
      abb83(29)=abb83(27)*abb83(9)
      abb83(27)=abb83(2)*abb83(27)*abb83(7)
      abb83(27)=abb83(29)+abb83(27)
      abb83(27)=mT*abb83(27)
      abb83(29)=spbl5k1*spak2l5
      abb83(29)=abb83(29)-abb83(28)
      abb83(29)=abb83(20)*abb83(29)
      abb83(9)=abb83(29)*abb83(9)
      abb83(7)=abb83(2)*abb83(29)*abb83(7)
      abb83(7)=abb83(27)+abb83(9)+abb83(7)
      abb83(7)=mT*abb83(7)
      abb83(9)=abb83(28)*abb83(8)
      abb83(7)=abb83(7)+abb83(9)
      abb83(7)=abb83(7)*abb83(14)
      abb83(9)=abb83(14)*abb83(24)*abb83(8)
      abb83(24)=abb83(25)*spal4l5*abb83(12)
      abb83(15)=abb83(15)*abb83(14)
      abb83(23)=abb83(23)*abb83(15)
      abb83(12)=-abb83(25)*spak1l5*abb83(12)
      abb83(20)=-abb83(20)*abb83(18)
      abb83(18)=mT*abb83(5)*abb83(18)
      abb83(18)=2.0_ki*abb83(20)+abb83(18)
      abb83(18)=mT*abb83(18)
      abb83(18)=abb83(18)+abb83(8)
      abb83(14)=abb83(18)*abb83(14)
      abb83(8)=abb83(15)*abb83(8)*abb83(21)*spbl5k1
      R2d83=0.0_ki
      rat2 = rat2 + R2d83
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='83' value='", &
          & R2d83, "'/>"
      end if
   end subroutine
end module p0_ubaru_httbar_abbrevd83h2_qp
