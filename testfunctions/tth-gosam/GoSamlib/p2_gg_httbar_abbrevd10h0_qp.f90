module     p2_gg_httbar_abbrevd10h0_qp
   use p2_gg_httbar_config, only: ki => ki_qp
   use p2_gg_httbar_kinematics_qp, only: epstensor
   use p2_gg_httbar_globalsh0_qp
   implicit none
   private
   complex(ki), dimension(31), public :: abb10
   complex(ki), public :: R2d10
   public :: init_abbrev
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
contains
   subroutine     init_abbrev()
      use p2_gg_httbar_config, only: deltaOS, &
     &    logfile, debug_nlo_diagrams
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_color_qp, only: TR
      use p2_gg_httbar_globalsl1_qp, only: epspow
      implicit none
      abb10(1)=1.0_ki/(-mT**2+es34)
      abb10(2)=NC**(-1)
      abb10(3)=es12**(-1)
      abb10(4)=spak2l3**(-1)
      abb10(5)=spbl3k2**(-1)
      abb10(6)=spbl4k2**(-1)
      abb10(7)=spbl5k2**(-1)
      abb10(8)=sqrt(mT**2)
      abb10(9)=spbe2e1*spae1e2*gs**4*i_*TR*e*gHT*abb10(2)*abb10(1)
      abb10(10)=abb10(9)*abb10(3)
      abb10(11)=mT**2
      abb10(12)=abb10(11)*abb10(10)
      abb10(13)=abb10(8)*abb10(10)
      abb10(14)=abb10(13)*mT
      abb10(15)=abb10(12)+abb10(14)
      abb10(16)=c1-c2
      abb10(15)=-abb10(15)*abb10(16)
      abb10(17)=abb10(15)*abb10(6)
      abb10(18)=-abb10(10)*abb10(16)
      abb10(19)=spak2l4*mH**2*abb10(5)*abb10(4)
      abb10(20)=abb10(19)*abb10(18)
      abb10(17)=abb10(17)+abb10(20)
      abb10(20)=spak1l5*spbk2k1
      abb10(21)=abb10(20)*abb10(17)
      abb10(22)=abb10(15)*abb10(7)
      abb10(23)=spak1l4*abb10(22)
      abb10(12)=-abb10(12)*abb10(16)
      abb10(24)=spbl3k2*abb10(6)
      abb10(25)=abb10(24)*abb10(7)
      abb10(26)=abb10(25)*spak1l3
      abb10(27)=abb10(12)*abb10(26)
      abb10(23)=abb10(27)+abb10(23)
      abb10(23)=spbk2k1*abb10(23)
      abb10(27)=spak2l5*spbl3k2
      abb10(28)=spbl3k1*spak1l5
      abb10(27)=abb10(27)-abb10(28)
      abb10(28)=abb10(18)*spal3l4
      abb10(29)=-abb10(28)*abb10(27)
      abb10(23)=abb10(29)+abb10(23)+abb10(21)
      abb10(23)=1.0_ki/4.0_ki*abb10(23)
      abb10(29)=abb10(8)*mT
      abb10(26)=abb10(26)*abb10(18)*abb10(29)**2
      abb10(30)=abb10(8)**2
      abb10(15)=-abb10(30)*abb10(15)
      abb10(31)=-spak1l4*abb10(7)*abb10(15)
      abb10(26)=abb10(26)+abb10(31)
      abb10(26)=spbk2k1*abb10(26)
      abb10(27)=spal3l4*abb10(27)
      abb10(19)=abb10(19)*abb10(20)
      abb10(19)=-abb10(19)+abb10(27)
      abb10(18)=-abb10(19)*abb10(30)*abb10(18)
      abb10(15)=-abb10(15)*abb10(20)*abb10(6)
      abb10(15)=abb10(15)+abb10(18)+abb10(26)
      abb10(15)=1.0_ki/2.0_ki*abb10(15)
      abb10(14)=-abb10(14)*abb10(16)
      abb10(18)=abb10(7)*abb10(14)*spal3l4
      abb10(19)=spbl3k2*abb10(18)
      abb10(19)=-2.0_ki*abb10(19)-abb10(21)
      abb10(10)=mT*abb10(10)
      abb10(10)=abb10(10)+abb10(13)
      abb10(10)=-abb10(16)*abb10(10)*abb10(8)
      abb10(13)=spak2l4*abb10(10)
      abb10(14)=abb10(14)*abb10(24)
      abb10(21)=spak2l3*abb10(14)
      abb10(13)=abb10(13)+abb10(21)
      abb10(21)=-spak1l4*abb10(10)
      abb10(24)=-spak1l3*abb10(14)
      abb10(21)=abb10(21)+abb10(24)
      abb10(9)=abb10(16)*abb10(9)
      abb10(16)=abb10(11)+abb10(29)
      abb10(16)=abb10(7)*abb10(16)*abb10(9)
      abb10(24)=-spak2l5*abb10(10)
      abb10(16)=1.0_ki/2.0_ki*abb10(16)+abb10(24)
      abb10(10)=spak1l5*abb10(10)
      abb10(24)=1.0_ki/2.0_ki*abb10(25)
      abb10(9)=abb10(24)*abb10(11)*abb10(9)
      abb10(11)=-spak2l5*abb10(14)
      abb10(9)=abb10(9)+abb10(11)
      abb10(11)=-abb10(12)*abb10(25)
      abb10(14)=spak1l5*abb10(14)
      abb10(25)=1.0_ki/2.0_ki*abb10(28)
      abb10(20)=-abb10(20)*abb10(25)
      abb10(26)=-spbk2k1*abb10(18)
      abb10(27)=spak2l5*spbk2k1
      abb10(28)=-abb10(25)*abb10(27)
      abb10(26)=abb10(26)+abb10(28)
      abb10(27)=-abb10(27)*abb10(17)
      abb10(18)=spbl3k1*abb10(18)
      abb10(18)=1.0_ki/2.0_ki*abb10(27)+abb10(18)
      abb10(27)=1.0_ki/2.0_ki*abb10(17)
      abb10(28)=1.0_ki/2.0_ki*abb10(22)
      abb10(12)=abb10(12)*abb10(24)
      R2d10=abb10(23)
      rat2 = rat2 + R2d10
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='10' value='", &
          & R2d10, "'/>"
      end if
   end subroutine
end module p2_gg_httbar_abbrevd10h0_qp
