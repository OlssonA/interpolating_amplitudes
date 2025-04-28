module     p2_gg_httbar_abbrevd75h4_qp
   use p2_gg_httbar_config, only: ki => ki_qp
   use p2_gg_httbar_kinematics_qp, only: epstensor
   use p2_gg_httbar_globalsh4_qp
   implicit none
   private
   complex(ki), dimension(54), public :: abb75
   complex(ki), public :: R2d75
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
      abb75(1)=1.0_ki/(mH**2-es34+es51-es23)
      abb75(2)=sqrt(mT**2)
      abb75(3)=NC**(-1)
      abb75(4)=spak2l3**(-1)
      abb75(5)=spbl3k2**(-1)
      abb75(6)=spak2l4**(-1)
      abb75(7)=spbl5k2**(-1)
      abb75(8)=spbl4k2**(-1)
      abb75(9)=abb75(2)**3
      abb75(10)=abb75(1)*c2*e*gHT*abb75(3)*gs**4*i_*TR
      abb75(11)=spbl4e2*abb75(10)
      abb75(12)=abb75(9)*abb75(11)
      abb75(13)=abb75(12)*spae1e2
      abb75(14)=abb75(2)**2
      abb75(15)=abb75(11)*mT
      abb75(16)=abb75(14)*abb75(15)
      abb75(17)=abb75(6)*spae2k2
      abb75(18)=abb75(16)*abb75(17)
      abb75(19)=abb75(18)*spae1l4
      abb75(13)=abb75(13)+abb75(19)
      abb75(19)=spak1l5*abb75(13)
      abb75(20)=abb75(12)*spae1l5
      abb75(21)=abb75(15)*abb75(7)
      abb75(22)=abb75(21)*abb75(14)
      abb75(23)=spbl3k2*spae1l3
      abb75(24)=abb75(23)*abb75(22)
      abb75(25)=-abb75(20)+abb75(24)
      abb75(25)=spak1e2*abb75(25)
      abb75(11)=abb75(11)*abb75(2)
      abb75(26)=abb75(11)*abb75(17)
      abb75(27)=mT**2
      abb75(28)=abb75(27)*abb75(7)
      abb75(29)=abb75(26)*abb75(28)
      abb75(30)=abb75(23)*abb75(29)
      abb75(18)=abb75(18)*spae1l5
      abb75(31)=-abb75(18)+abb75(30)
      abb75(31)=spak1l4*abb75(31)
      abb75(19)=abb75(31)+abb75(25)+abb75(19)
      abb75(19)=spbk1e1*abb75(19)
      abb75(25)=abb75(11)*spae2k2
      abb75(31)=spbk2k1*spae1k1
      abb75(32)=abb75(25)*abb75(31)
      abb75(33)=abb75(13)-abb75(32)
      abb75(34)=spal3l5*abb75(33)
      abb75(35)=spae2l3*abb75(20)
      abb75(36)=-spal3l4*abb75(18)
      abb75(34)=abb75(36)+abb75(35)+abb75(34)
      abb75(34)=spbl3e1*abb75(34)
      abb75(10)=abb75(17)*abb75(10)
      abb75(35)=abb75(28)*abb75(10)
      abb75(9)=abb75(35)*abb75(9)
      abb75(36)=spbk2e1*abb75(9)
      abb75(37)=-spbl3e2*abb75(36)
      abb75(38)=abb75(21)*spae2l4
      abb75(39)=abb75(38)*spbk2e1
      abb75(40)=abb75(39)*abb75(14)
      abb75(41)=-spbl4l3*abb75(40)
      abb75(37)=abb75(41)+abb75(37)
      abb75(37)=spae1l3*abb75(37)
      abb75(16)=abb75(12)-abb75(16)
      abb75(41)=mH**2*abb75(5)*abb75(4)
      abb75(42)=abb75(41)*spbk2e1
      abb75(16)=abb75(16)*abb75(42)
      abb75(12)=abb75(12)*spbk2e1
      abb75(16)=-abb75(12)+abb75(16)
      abb75(16)=spae1l5*spae2k2*abb75(16)
      abb75(10)=abb75(10)*mT
      abb75(43)=abb75(2)**4
      abb75(44)=-spae1l5*abb75(43)*abb75(10)
      abb75(45)=-abb75(31)*abb75(9)
      abb75(44)=abb75(44)+abb75(45)
      abb75(44)=spbe2e1*abb75(44)
      abb75(45)=spae2l4*abb75(20)
      abb75(46)=abb75(31)*abb75(38)
      abb75(47)=abb75(14)*abb75(46)
      abb75(45)=abb75(45)+abb75(47)
      abb75(45)=spbl4e1*abb75(45)
      abb75(47)=abb75(42)*spak2l5
      abb75(33)=abb75(33)*abb75(47)
      abb75(48)=spbl3e1*spal3l5
      abb75(47)=abb75(47)+abb75(48)
      abb75(26)=abb75(26)*abb75(27)
      abb75(49)=abb75(26)*abb75(31)*abb75(47)
      abb75(12)=abb75(17)*abb75(12)
      abb75(27)=spae1l5*abb75(27)*abb75(12)
      abb75(27)=abb75(27)+abb75(49)
      abb75(27)=abb75(8)*abb75(27)
      abb75(49)=abb75(38)*spbl4e1
      abb75(50)=abb75(49)*abb75(14)
      abb75(9)=abb75(9)*spbe2e1
      abb75(9)=abb75(50)-abb75(9)
      abb75(23)=-abb75(9)*abb75(23)
      abb75(10)=abb75(10)*abb75(14)
      abb75(14)=abb75(10)*spae1k1
      abb75(50)=-abb75(14)*abb75(47)
      abb75(51)=spae1k1*abb75(36)
      abb75(50)=abb75(51)+abb75(50)
      abb75(50)=spbe2k1*abb75(50)
      abb75(51)=abb75(11)*spae2l4
      abb75(52)=abb75(51)*spae1k1
      abb75(53)=abb75(52)*abb75(47)
      abb75(54)=-spae1k1*abb75(40)
      abb75(53)=abb75(54)+abb75(53)
      abb75(53)=spbl4k1*abb75(53)
      abb75(21)=-spae1e2*abb75(43)*abb75(21)*spbk2e1
      abb75(12)=-spae1l4*abb75(28)*abb75(12)
      abb75(12)=abb75(53)+abb75(50)+abb75(23)+abb75(27)+abb75(33)+abb75(12)+abb&
      &75(45)+abb75(21)+abb75(16)+abb75(44)+abb75(34)+abb75(37)+abb75(19)
      abb75(16)=abb75(22)*spae1e2
      abb75(19)=abb75(29)*spae1l4
      abb75(16)=abb75(16)+abb75(19)
      abb75(19)=abb75(26)*abb75(8)
      abb75(21)=spae1l5*abb75(19)
      abb75(23)=abb75(35)*abb75(2)
      abb75(26)=abb75(23)*spbe2k1
      abb75(27)=spae1k1*abb75(26)
      abb75(28)=abb75(11)*spae1l5
      abb75(33)=-spae2k2*abb75(28)
      abb75(21)=abb75(27)+abb75(21)+abb75(33)-abb75(16)
      abb75(21)=spbk2e1*abb75(21)
      abb75(15)=abb75(15)*abb75(17)
      abb75(17)=abb75(15)*spae1l4
      abb75(11)=abb75(11)*spae1e2
      abb75(11)=abb75(11)+abb75(17)
      abb75(17)=abb75(11)*spak1l5
      abb75(15)=abb75(15)*spae1l5
      abb75(27)=abb75(15)*spak1l4
      abb75(33)=abb75(28)*spak1e2
      abb75(17)=abb75(17)-abb75(33)-abb75(27)
      abb75(27)=spbk1e1*abb75(17)
      abb75(33)=-spae1l5*abb75(10)
      abb75(34)=-abb75(23)*abb75(31)
      abb75(33)=abb75(33)+abb75(34)
      abb75(33)=spbe2e1*abb75(33)
      abb75(34)=spae1l5*abb75(51)
      abb75(34)=abb75(34)+abb75(46)
      abb75(34)=spbl4e1*abb75(34)
      abb75(35)=abb75(39)*spbl4k1
      abb75(37)=-spae1k1*abb75(35)
      abb75(21)=abb75(37)+abb75(34)+abb75(33)+abb75(21)+abb75(27)
      abb75(20)=-3.0_ki*abb75(20)+abb75(24)
      abb75(24)=-abb75(10)*abb75(47)
      abb75(24)=3.0_ki*abb75(36)+abb75(24)
      abb75(27)=abb75(23)*spbk2e1
      abb75(31)=abb75(31)*abb75(19)
      abb75(14)=spbe2k1*abb75(14)
      abb75(33)=spbl4k1*abb75(52)
      abb75(14)=-abb75(33)+abb75(14)+abb75(32)-abb75(31)
      abb75(13)=3.0_ki*abb75(13)-2.0_ki*abb75(14)
      abb75(14)=-2.0_ki*abb75(10)
      abb75(31)=abb75(51)*abb75(47)
      abb75(31)=-3.0_ki*abb75(40)+abb75(31)
      abb75(32)=2.0_ki*abb75(51)
      abb75(18)=-3.0_ki*abb75(18)+abb75(30)
      abb75(10)=abb75(10)*spbe2e1
      abb75(30)=abb75(51)*spbl4e1
      abb75(10)=abb75(10)-abb75(30)
      abb75(19)=abb75(19)-abb75(25)
      abb75(25)=-spbk2e1*abb75(19)
      abb75(25)=abb75(25)+abb75(10)
      abb75(25)=spal3l5*abb75(25)
      abb75(30)=abb75(23)*spbe2e1
      abb75(30)=abb75(30)-abb75(49)
      abb75(33)=-spbk2k1*abb75(30)
      abb75(34)=spbk2e1*abb75(26)
      abb75(33)=-abb75(35)+abb75(34)+abb75(33)
      abb75(33)=spak1l3*abb75(33)
      abb75(25)=abb75(33)+abb75(25)
      abb75(16)=-spbl3k2*abb75(16)
      abb75(33)=spbl3k1*abb75(17)
      abb75(16)=abb75(16)+abb75(33)
      abb75(22)=2.0_ki*abb75(22)
      abb75(33)=-spak1e2*abb75(22)
      abb75(29)=2.0_ki*abb75(29)
      abb75(34)=-spak1l4*abb75(29)
      abb75(33)=abb75(34)+abb75(33)
      abb75(33)=spbk1e1*abb75(33)
      abb75(34)=abb75(48)*abb75(19)
      abb75(10)=spak2l5*abb75(41)*abb75(10)
      abb75(35)=-spbl4k1*abb75(38)
      abb75(26)=abb75(26)+abb75(35)
      abb75(26)=spak1k2*abb75(42)*abb75(26)
      abb75(35)=abb75(30)*abb75(41)
      abb75(36)=-es12*abb75(35)
      abb75(9)=abb75(36)+abb75(26)+abb75(10)+3.0_ki*abb75(9)+abb75(34)+abb75(33)
      abb75(10)=2.0_ki*abb75(35)
      abb75(19)=2.0_ki*abb75(19)
      abb75(23)=-abb75(42)*abb75(23)
      abb75(17)=abb75(17)*abb75(41)*spbk2k1
      abb75(26)=abb75(41)*abb75(28)
      abb75(33)=-abb75(41)*abb75(11)
      abb75(34)=abb75(41)*abb75(15)
      abb75(35)=abb75(38)*abb75(42)
      R2d75=0.0_ki
      rat2 = rat2 + R2d75
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='75' value='", &
          & R2d75, "'/>"
      end if
   end subroutine
end module p2_gg_httbar_abbrevd75h4_qp
