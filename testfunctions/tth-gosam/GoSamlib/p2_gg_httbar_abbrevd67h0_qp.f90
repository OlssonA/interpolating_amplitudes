module     p2_gg_httbar_abbrevd67h0_qp
   use p2_gg_httbar_config, only: ki => ki_qp
   use p2_gg_httbar_kinematics_qp, only: epstensor
   use p2_gg_httbar_globalsh0_qp
   implicit none
   private
   complex(ki), dimension(48), public :: abb67
   complex(ki), public :: R2d67
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
      abb67(1)=sqrt(mT**2)
      abb67(2)=NC**(-1)
      abb67(3)=es12**(-1)
      abb67(4)=spbl4k2**(-1)
      abb67(5)=spbl5k2**(-1)
      abb67(6)=spak2l3**(-1)
      abb67(7)=spbl3k2**(-1)
      abb67(8)=spak2l5**(-1)
      abb67(9)=spak2l4**(-1)
      abb67(10)=spak2l5*spbl5k2
      abb67(11)=spbl3k2*spal3l5
      abb67(12)=abb67(10)*abb67(11)
      abb67(13)=spbl3k1*spal3l5
      abb67(14)=spbl5k2*abb67(13)*spak1l5
      abb67(12)=abb67(14)-abb67(12)
      abb67(12)=abb67(12)*abb67(4)
      abb67(14)=abb67(4)*spak1l5
      abb67(15)=abb67(14)*spbl5k2
      abb67(16)=mH**2*abb67(6)*abb67(7)
      abb67(17)=abb67(16)*spak2l5
      abb67(18)=abb67(15)*abb67(17)
      abb67(17)=abb67(17)*spak1l4
      abb67(18)=abb67(18)+abb67(17)
      abb67(18)=abb67(18)*spbk2k1
      abb67(19)=abb67(5)*spbl3k2
      abb67(20)=abb67(19)*spak1l3
      abb67(21)=spal4l5*spbl5k1
      abb67(22)=abb67(20)*abb67(21)
      abb67(23)=spak2l3*spal4l5*spbl3k2
      abb67(24)=abb67(13)*spak1l4
      abb67(25)=spak2l4*spbl3k2
      abb67(26)=abb67(25)*spal3l5
      abb67(12)=-abb67(26)+abb67(12)+abb67(23)+abb67(18)+abb67(24)-abb67(22)
      abb67(18)=c2-c1
      abb67(12)=abb67(12)*abb67(18)
      abb67(22)=-spbk2k1*abb67(18)
      abb67(23)=mT**2
      abb67(24)=abb67(22)*abb67(23)
      abb67(27)=abb67(5)**2
      abb67(28)=abb67(27)*abb67(25)*abb67(8)*spak1l3
      abb67(29)=abb67(20)*abb67(4)
      abb67(28)=abb67(28)+abb67(29)
      abb67(28)=-abb67(28)*abb67(24)
      abb67(12)=abb67(28)+abb67(12)
      abb67(12)=mT*abb67(12)
      abb67(28)=spbl3k1*spal3l4
      abb67(28)=abb67(28)+abb67(21)
      abb67(28)=abb67(28)*spak1l5
      abb67(29)=spak2l4*spbl4k2
      abb67(10)=abb67(29)+abb67(10)
      abb67(10)=abb67(10)*spal4l5
      abb67(29)=spal4l5*spbl4k1
      abb67(29)=abb67(29)-abb67(13)
      abb67(29)=abb67(29)*spak1l4
      abb67(30)=abb67(16)*spak1l5
      abb67(31)=abb67(30)*spak2l4
      abb67(17)=abb67(31)-abb67(17)
      abb67(17)=abb67(17)*spbk2k1
      abb67(31)=spak2l5*spbl3k2
      abb67(32)=abb67(31)*spal3l4
      abb67(10)=-abb67(28)-abb67(17)+abb67(32)-abb67(26)+abb67(10)-abb67(29)
      abb67(10)=abb67(18)*abb67(10)
      abb67(17)=abb67(4)*spak2l5
      abb67(26)=abb67(17)*abb67(9)
      abb67(26)=abb67(26)+abb67(5)
      abb67(26)=abb67(26)*spak1l4
      abb67(28)=spak2l4*abb67(8)
      abb67(29)=abb67(28)*abb67(5)
      abb67(32)=abb67(29)*spak1l5
      abb67(32)=abb67(26)-abb67(32)-abb67(14)
      abb67(33)=abb67(32)*abb67(24)
      abb67(34)=spak1l4*abb67(5)
      abb67(35)=abb67(34)+abb67(14)
      abb67(36)=abb67(1)*mT
      abb67(35)=-abb67(35)*abb67(22)*abb67(36)
      abb67(10)=abb67(10)+abb67(33)+abb67(35)
      abb67(33)=abb67(1)*abb67(10)
      abb67(12)=abb67(12)+abb67(33)
      abb67(33)=1.0_ki/2.0_ki*abb67(3)
      abb67(35)=e*gs**4*abb67(2)*gHT*spbe2e1*spae1e2*TR*i_
      abb67(33)=abb67(33)*abb67(35)
      abb67(37)=abb67(33)*abb67(1)
      abb67(12)=abb67(12)*abb67(37)
      abb67(14)=-abb67(26)+abb67(14)
      abb67(14)=abb67(16)*abb67(14)
      abb67(26)=abb67(29)*abb67(30)
      abb67(14)=abb67(26)+abb67(14)
      abb67(14)=abb67(14)*abb67(24)
      abb67(26)=abb67(18)*abb67(36)
      abb67(29)=abb67(16)*spbk2k1
      abb67(38)=abb67(29)*abb67(34)
      abb67(11)=abb67(11)*abb67(4)
      abb67(11)=abb67(38)+abb67(11)+2.0_ki*spal4l5
      abb67(11)=-abb67(11)*abb67(26)
      abb67(11)=abb67(14)+abb67(11)
      abb67(14)=abb67(35)*abb67(3)
      abb67(11)=abb67(11)*abb67(14)
      abb67(10)=abb67(10)*abb67(33)
      abb67(22)=abb67(22)*abb67(33)
      abb67(38)=abb67(36)*abb67(4)*spak1l3*abb67(22)
      abb67(39)=-abb67(4)*abb67(18)
      abb67(40)=abb67(36)*abb67(16)*abb67(39)
      abb67(41)=abb67(18)*mT
      abb67(42)=abb67(17)*spbl5k2
      abb67(42)=abb67(42)+spak2l4
      abb67(42)=-abb67(42)*abb67(41)
      abb67(43)=3.0_ki/2.0_ki*abb67(1)
      abb67(44)=abb67(18)*spak2l4
      abb67(45)=-abb67(44)*abb67(43)
      abb67(42)=abb67(42)+abb67(45)
      abb67(42)=abb67(42)*abb67(3)*abb67(1)
      abb67(40)=1.0_ki/2.0_ki*abb67(40)+abb67(42)
      abb67(35)=abb67(40)*abb67(35)
      abb67(36)=abb67(14)*abb67(36)
      abb67(39)=abb67(39)*abb67(36)
      abb67(40)=2.0_ki*abb67(39)
      abb67(42)=-abb67(44)*abb67(33)
      abb67(15)=abb67(15)+spak1l4
      abb67(15)=abb67(15)*abb67(41)
      abb67(44)=abb67(18)*spak1l4
      abb67(43)=abb67(44)*abb67(43)
      abb67(15)=abb67(15)+abb67(43)
      abb67(14)=abb67(15)*abb67(1)*abb67(14)
      abb67(15)=abb67(44)*abb67(33)
      abb67(19)=abb67(19)*spak2l3*abb67(41)
      abb67(43)=3.0_ki*abb67(1)
      abb67(44)=abb67(18)*spak2l5
      abb67(45)=abb67(44)*abb67(43)
      abb67(19)=abb67(19)+abb67(45)
      abb67(19)=abb67(19)*abb67(37)
      abb67(45)=abb67(36)*abb67(18)*abb67(5)
      abb67(46)=-2.0_ki*abb67(45)
      abb67(44)=abb67(44)*abb67(33)
      abb67(20)=-abb67(20)*abb67(41)
      abb67(41)=abb67(18)*spak1l5
      abb67(43)=-abb67(41)*abb67(43)
      abb67(20)=abb67(20)+abb67(43)
      abb67(20)=abb67(20)*abb67(37)
      abb67(37)=-abb67(41)*abb67(33)
      abb67(41)=spak1l5*spbl3k1
      abb67(31)=abb67(31)-abb67(41)
      abb67(41)=abb67(18)*abb67(33)
      abb67(31)=-abb67(41)*spal4l5*abb67(31)
      abb67(43)=abb67(25)*spal4l5
      abb67(47)=spak1l4*spal4l5
      abb67(48)=abb67(47)*spbl3k1
      abb67(43)=abb67(43)-abb67(48)
      abb67(41)=-abb67(43)*abb67(41)
      abb67(23)=abb67(23)*abb67(18)*spbl3k1*abb67(32)
      abb67(25)=abb67(25)*abb67(5)
      abb67(32)=abb67(34)*spbl3k1
      abb67(25)=abb67(25)-abb67(32)
      abb67(25)=abb67(25)*abb67(26)
      abb67(23)=abb67(23)+abb67(25)
      abb67(23)=abb67(23)*abb67(33)
      abb67(25)=-abb67(30)*spal4l5*abb67(22)
      abb67(16)=-abb67(47)*abb67(16)*abb67(22)
      abb67(22)=abb67(27)*abb67(28)
      abb67(26)=abb67(4)*abb67(5)
      abb67(22)=abb67(26)+abb67(22)
      abb67(22)=abb67(22)*abb67(24)
      abb67(17)=abb67(29)*abb67(17)
      abb67(13)=abb67(13)*abb67(4)
      abb67(13)=abb67(13)+abb67(17)
      abb67(17)=abb67(21)*abb67(5)
      abb67(13)=abb67(17)+1.0_ki/2.0_ki*abb67(13)
      abb67(13)=abb67(13)*abb67(18)
      abb67(13)=abb67(22)+abb67(13)
      abb67(13)=abb67(13)*abb67(36)
      R2d67=0.0_ki
      rat2 = rat2 + R2d67
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='67' value='", &
          & R2d67, "'/>"
      end if
   end subroutine
end module p2_gg_httbar_abbrevd67h0_qp
