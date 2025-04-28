module     p2_gg_httbar_abbrevd12h0
   use p2_gg_httbar_config, only: ki
   use p2_gg_httbar_kinematics, only: epstensor
   use p2_gg_httbar_globalsh0
   implicit none
   private
   complex(ki), dimension(31), public :: abb12
   complex(ki), public :: R2d12
   public :: init_abbrev
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
contains
   subroutine     init_abbrev()
      use p2_gg_httbar_config, only: deltaOS, &
     &    logfile, debug_nlo_diagrams
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_model
      use p2_gg_httbar_color, only: TR
      use p2_gg_httbar_globalsl1, only: epspow
      implicit none
      abb12(1)=1.0_ki/(mH**2+mT**2-es34-es45+es12)
      abb12(2)=NC**(-1)
      abb12(3)=es12**(-1)
      abb12(4)=spbl4k2**(-1)
      abb12(5)=spak2l3**(-1)
      abb12(6)=spbl3k2**(-1)
      abb12(7)=spbl5k2**(-1)
      abb12(8)=sqrt(mT**2)
      abb12(9)=spbe2e1*spae1e2*gs**4*i_*TR*e*gHT*abb12(2)*abb12(1)
      abb12(10)=abb12(9)*abb12(3)
      abb12(11)=mT**2
      abb12(12)=abb12(11)*abb12(10)
      abb12(13)=abb12(8)*abb12(10)
      abb12(14)=abb12(13)*mT
      abb12(15)=abb12(12)+abb12(14)
      abb12(16)=c1-c2
      abb12(15)=-abb12(15)*abb12(16)
      abb12(17)=abb12(15)*abb12(7)
      abb12(18)=-abb12(10)*abb12(16)
      abb12(19)=spak2l5*mH**2*abb12(6)*abb12(5)
      abb12(20)=abb12(19)*abb12(18)
      abb12(17)=abb12(17)+abb12(20)
      abb12(20)=spak1l4*spbk2k1
      abb12(21)=abb12(20)*abb12(17)
      abb12(22)=abb12(15)*abb12(4)
      abb12(23)=spak1l5*abb12(22)
      abb12(12)=-abb12(12)*abb12(16)
      abb12(24)=spbl3k2*abb12(7)
      abb12(25)=abb12(24)*abb12(4)
      abb12(26)=abb12(25)*spak1l3
      abb12(27)=abb12(12)*abb12(26)
      abb12(23)=abb12(27)+abb12(23)
      abb12(23)=spbk2k1*abb12(23)
      abb12(27)=spak2l4*spbl3k2
      abb12(28)=spbl3k1*spak1l4
      abb12(27)=abb12(27)-abb12(28)
      abb12(28)=abb12(18)*spal3l5
      abb12(29)=-abb12(28)*abb12(27)
      abb12(23)=abb12(29)+abb12(23)+abb12(21)
      abb12(23)=1.0_ki/4.0_ki*abb12(23)
      abb12(29)=abb12(8)*mT
      abb12(26)=abb12(26)*abb12(18)*abb12(29)**2
      abb12(30)=abb12(8)**2
      abb12(15)=-abb12(30)*abb12(15)
      abb12(31)=-spak1l5*abb12(4)*abb12(15)
      abb12(26)=abb12(26)+abb12(31)
      abb12(26)=spbk2k1*abb12(26)
      abb12(27)=spal3l5*abb12(27)
      abb12(19)=abb12(19)*abb12(20)
      abb12(19)=-abb12(19)+abb12(27)
      abb12(18)=-abb12(19)*abb12(30)*abb12(18)
      abb12(15)=-abb12(15)*abb12(20)*abb12(7)
      abb12(15)=abb12(15)+abb12(18)+abb12(26)
      abb12(15)=1.0_ki/2.0_ki*abb12(15)
      abb12(14)=-abb12(14)*abb12(16)
      abb12(18)=abb12(4)*abb12(14)*spal3l5
      abb12(19)=spbl3k2*abb12(18)
      abb12(19)=-2.0_ki*abb12(19)-abb12(21)
      abb12(9)=abb12(16)*abb12(9)
      abb12(21)=abb12(11)+abb12(29)
      abb12(21)=abb12(4)*abb12(21)*abb12(9)
      abb12(10)=mT*abb12(10)
      abb12(10)=abb12(10)+abb12(13)
      abb12(10)=-abb12(16)*abb12(10)*abb12(8)
      abb12(13)=-spak2l4*abb12(10)
      abb12(13)=1.0_ki/2.0_ki*abb12(21)+abb12(13)
      abb12(16)=spak1l4*abb12(10)
      abb12(21)=spak2l5*abb12(10)
      abb12(14)=abb12(14)*abb12(24)
      abb12(24)=spak2l3*abb12(14)
      abb12(21)=abb12(21)+abb12(24)
      abb12(10)=-spak1l5*abb12(10)
      abb12(24)=-spak1l3*abb12(14)
      abb12(10)=abb12(10)+abb12(24)
      abb12(24)=1.0_ki/2.0_ki*abb12(25)
      abb12(9)=abb12(24)*abb12(11)*abb12(9)
      abb12(11)=-spak2l4*abb12(14)
      abb12(9)=abb12(9)+abb12(11)
      abb12(11)=-abb12(12)*abb12(25)
      abb12(14)=spak1l4*abb12(14)
      abb12(25)=1.0_ki/2.0_ki*abb12(28)
      abb12(20)=-abb12(20)*abb12(25)
      abb12(26)=-spbk2k1*abb12(18)
      abb12(27)=spak2l4*spbk2k1
      abb12(28)=-abb12(25)*abb12(27)
      abb12(26)=abb12(26)+abb12(28)
      abb12(27)=-abb12(27)*abb12(17)
      abb12(18)=spbl3k1*abb12(18)
      abb12(18)=1.0_ki/2.0_ki*abb12(27)+abb12(18)
      abb12(27)=1.0_ki/2.0_ki*abb12(22)
      abb12(28)=1.0_ki/2.0_ki*abb12(17)
      abb12(12)=abb12(12)*abb12(24)
      R2d12=abb12(23)
      rat2 = rat2 + R2d12
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='12' value='", &
          & R2d12, "'/>"
      end if
   end subroutine
end module p2_gg_httbar_abbrevd12h0
