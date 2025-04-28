module     p2_gg_httbar_abbrevd73h0_qp
   use p2_gg_httbar_config, only: ki => ki_qp
   use p2_gg_httbar_kinematics_qp, only: epstensor
   use p2_gg_httbar_globalsh0_qp
   implicit none
   private
   complex(ki), dimension(55), public :: abb73
   complex(ki), public :: R2d73
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
      abb73(1)=1.0_ki/(-mT**2+es51)
      abb73(2)=sqrt(mT**2)
      abb73(3)=NC**(-1)
      abb73(4)=spbl5k2**(-1)
      abb73(5)=spak2l3**(-1)
      abb73(6)=spbl3k2**(-1)
      abb73(7)=spbl4k2**(-1)
      abb73(8)=spak2l5**(-1)
      abb73(9)=abb73(1)*c2*e*gHT*abb73(3)*gs**4*i_*TR
      abb73(10)=spae1l5*abb73(9)
      abb73(11)=abb73(4)*spbk2e1
      abb73(12)=abb73(10)*abb73(11)
      abb73(13)=abb73(12)*abb73(8)
      abb73(14)=mT*abb73(2)
      abb73(15)=abb73(14)**2
      abb73(16)=abb73(13)*abb73(15)
      abb73(17)=spak2l3*abb73(16)
      abb73(9)=abb73(11)*abb73(9)
      abb73(18)=mT*abb73(2)**3
      abb73(19)=abb73(18)*abb73(9)
      abb73(20)=-spae1l3*abb73(19)
      abb73(17)=abb73(20)+abb73(17)
      abb73(17)=spae2l4*abb73(17)
      abb73(20)=spal3l5*spbl5e1
      abb73(21)=spak1l3*spbk1e1
      abb73(20)=abb73(20)+abb73(21)
      abb73(21)=abb73(2)**2
      abb73(22)=abb73(10)*spae2l4
      abb73(23)=abb73(21)*abb73(22)
      abb73(24)=-abb73(23)*abb73(20)
      abb73(25)=spae2l5*spbl5e1
      abb73(26)=spak1e2*spbk1e1
      abb73(25)=abb73(25)+abb73(26)
      abb73(26)=abb73(21)*abb73(10)
      abb73(27)=abb73(25)*abb73(26)
      abb73(28)=abb73(19)*spae1e2
      abb73(27)=abb73(27)+abb73(28)
      abb73(28)=spae2k2*abb73(16)
      abb73(28)=abb73(28)+abb73(27)
      abb73(28)=spal3l4*abb73(28)
      abb73(17)=abb73(28)+abb73(24)+abb73(17)
      abb73(17)=spbl3e2*abb73(17)
      abb73(24)=abb73(10)*spbk2e2
      abb73(21)=abb73(21)*abb73(24)
      abb73(28)=abb73(21)*spak2l4
      abb73(29)=abb73(24)*abb73(7)
      abb73(30)=abb73(18)*abb73(29)
      abb73(31)=-abb73(30)-abb73(28)
      abb73(31)=spbl3e1*abb73(31)
      abb73(32)=abb73(24)*spak2l4
      abb73(33)=abb73(11)*abb73(32)*abb73(14)
      abb73(34)=abb73(29)*abb73(11)
      abb73(35)=abb73(34)*abb73(15)
      abb73(36)=abb73(35)+abb73(33)
      abb73(36)=spbl5l3*abb73(36)
      abb73(37)=abb73(9)*spae1k1
      abb73(38)=abb73(15)*abb73(37)
      abb73(39)=spbk2e2*abb73(7)
      abb73(40)=abb73(38)*abb73(39)
      abb73(41)=abb73(37)*abb73(14)
      abb73(42)=spak2l4*spbk2e2
      abb73(43)=abb73(42)*abb73(41)
      abb73(44)=abb73(40)+abb73(43)
      abb73(44)=spbl3k1*abb73(44)
      abb73(31)=abb73(44)+abb73(36)+abb73(31)
      abb73(31)=spae2l3*abb73(31)
      abb73(36)=mH**2*abb73(6)*abb73(5)
      abb73(44)=abb73(36)+1.0_ki
      abb73(44)=abb73(44)*spak2l4
      abb73(45)=abb73(21)*abb73(44)
      abb73(45)=abb73(45)+abb73(30)
      abb73(45)=abb73(45)*abb73(25)
      abb73(46)=spak2l5*spbl5e1
      abb73(47)=spak1k2*spbk1e1
      abb73(46)=abb73(46)+abb73(47)
      abb73(47)=abb73(36)*spae2l4
      abb73(48)=abb73(47)+spae2l4
      abb73(21)=-abb73(46)*abb73(48)*abb73(21)
      abb73(48)=abb73(39)*abb73(9)
      abb73(49)=abb73(2)**4
      abb73(50)=mT**2
      abb73(51)=abb73(49)*abb73(50)*abb73(48)
      abb73(44)=abb73(19)*spbk2e2*abb73(44)
      abb73(44)=abb73(51)+abb73(44)
      abb73(44)=spae1e2*abb73(44)
      abb73(49)=-abb73(49)*abb73(22)
      abb73(51)=abb73(10)*abb73(7)
      abb73(52)=abb73(18)*abb73(51)
      abb73(53)=spbl3k2*spae2l3
      abb73(54)=-abb73(53)*abb73(52)
      abb73(49)=abb73(49)+abb73(54)
      abb73(49)=spbe2e1*abb73(49)
      abb73(11)=abb73(51)*abb73(11)
      abb73(54)=abb73(15)*abb73(11)*abb73(53)
      abb73(18)=abb73(18)*spae2l4
      abb73(55)=abb73(12)*abb73(18)
      abb73(54)=abb73(55)+abb73(54)
      abb73(54)=spbl5e2*abb73(54)
      abb73(18)=abb73(37)*abb73(18)
      abb73(53)=abb73(7)*abb73(38)*abb73(53)
      abb73(18)=abb73(18)+abb73(53)
      abb73(18)=spbe2k1*abb73(18)
      abb73(17)=abb73(18)+abb73(54)+abb73(49)+abb73(44)+abb73(31)+abb73(21)+abb&
      &73(45)+abb73(17)
      abb73(18)=-abb73(24)*abb73(46)
      abb73(12)=abb73(12)*abb73(14)
      abb73(21)=spbl5e2*abb73(12)
      abb73(24)=spbe2k1*abb73(41)
      abb73(18)=abb73(24)+abb73(21)+abb73(18)
      abb73(18)=spae2l4*abb73(18)
      abb73(21)=abb73(29)*abb73(14)
      abb73(24)=abb73(32)+abb73(21)
      abb73(24)=abb73(24)*abb73(25)
      abb73(29)=abb73(48)*abb73(15)
      abb73(31)=abb73(9)*abb73(14)
      abb73(32)=abb73(31)*abb73(42)
      abb73(32)=abb73(29)+abb73(32)
      abb73(32)=spae1e2*abb73(32)
      abb73(42)=-spbe2e1*abb73(23)
      abb73(18)=abb73(42)+abb73(32)+abb73(18)+abb73(24)
      abb73(24)=-3.0_ki*abb73(30)-2.0_ki*abb73(28)
      abb73(28)=3.0_ki*spae2l4
      abb73(19)=-abb73(19)*abb73(28)
      abb73(28)=abb73(31)*spae2l4
      abb73(30)=-3.0_ki*abb73(35)-2.0_ki*abb73(33)
      abb73(32)=abb73(34)*abb73(50)
      abb73(33)=3.0_ki*abb73(23)
      abb73(35)=spbl5e1*abb73(33)
      abb73(42)=abb73(22)*spbl5e1
      abb73(44)=abb73(10)*abb73(14)
      abb73(44)=abb73(26)-abb73(44)
      abb73(44)=abb73(44)*spbk2e1
      abb73(45)=abb73(41)*spbk2k1
      abb73(44)=abb73(44)-abb73(45)
      abb73(45)=abb73(36)*abb73(44)
      abb73(16)=3.0_ki*abb73(16)
      abb73(45)=-abb73(16)+abb73(45)
      abb73(45)=spae2k2*abb73(45)
      abb73(46)=abb73(41)*spbl3k1
      abb73(48)=abb73(12)*spbl5l3
      abb73(46)=abb73(46)+abb73(48)
      abb73(48)=spbl3e1*abb73(26)
      abb73(48)=abb73(48)-abb73(46)
      abb73(48)=spae2l3*abb73(48)
      abb73(27)=-3.0_ki*abb73(27)+abb73(48)+abb73(45)
      abb73(25)=abb73(10)*abb73(25)
      abb73(45)=abb73(31)*spae1e2
      abb73(13)=abb73(13)*abb73(50)
      abb73(48)=abb73(13)*spae2k2
      abb73(25)=abb73(45)+abb73(48)+abb73(25)
      abb73(26)=2.0_ki*abb73(26)
      abb73(12)=2.0_ki*abb73(12)
      abb73(45)=abb73(2)-mT
      abb73(10)=spbk2e1*mT*abb73(39)*abb73(10)*abb73(45)
      abb73(45)=abb73(2)*abb73(8)*mT**3
      abb73(34)=-abb73(34)*abb73(45)
      abb73(37)=abb73(37)*abb73(50)
      abb73(39)=abb73(37)*abb73(39)
      abb73(48)=-spbk2k1*abb73(39)
      abb73(10)=abb73(48)+abb73(10)+abb73(34)
      abb73(10)=spak2l3*abb73(10)
      abb73(34)=abb73(21)*abb73(20)
      abb73(29)=spae1l3*abb73(29)
      abb73(10)=abb73(29)+abb73(34)+abb73(10)
      abb73(29)=-spak2l4*abb73(25)
      abb73(34)=spak2l5*abb73(42)
      abb73(22)=abb73(22)*spbk1e1
      abb73(48)=spak1k2*abb73(22)
      abb73(29)=abb73(48)+abb73(34)+abb73(29)
      abb73(29)=spbl3k2*abb73(29)
      abb73(34)=spae2l4*abb73(46)
      abb73(23)=-spbl3e1*abb73(23)
      abb73(23)=abb73(23)+abb73(29)+abb73(34)
      abb73(14)=abb73(51)*abb73(14)
      abb73(20)=-abb73(14)*abb73(20)
      abb73(29)=abb73(45)*abb73(11)
      abb73(34)=spak2l3*abb73(29)
      abb73(9)=abb73(9)*abb73(15)*abb73(7)
      abb73(45)=-spae1l3*abb73(9)
      abb73(20)=abb73(45)+abb73(34)+abb73(20)
      abb73(20)=spbl3e2*abb73(20)
      abb73(34)=-spbe2e1*abb73(52)
      abb73(11)=abb73(11)*spbl5e2
      abb73(15)=abb73(15)*abb73(11)
      abb73(45)=spbe2k1*abb73(7)
      abb73(38)=abb73(38)*abb73(45)
      abb73(15)=abb73(38)+abb73(34)+abb73(15)
      abb73(15)=3.0_ki*abb73(15)+abb73(20)
      abb73(20)=abb73(14)*spbe2e1
      abb73(11)=abb73(11)*abb73(50)
      abb73(34)=abb73(45)*abb73(37)
      abb73(11)=-abb73(20)+abb73(11)+abb73(34)
      abb73(20)=2.0_ki*abb73(36)*abb73(11)
      abb73(9)=-2.0_ki*abb73(9)
      abb73(14)=2.0_ki*abb73(14)
      abb73(34)=spbl5e1*abb73(14)
      abb73(37)=-abb73(47)*abb73(44)
      abb73(16)=spae2l4*abb73(16)
      abb73(16)=abb73(16)+abb73(37)
      abb73(37)=2.0_ki*abb73(13)*abb73(47)
      abb73(13)=abb73(13)*spae2l4
      abb73(29)=2.0_ki*abb73(29)
      abb73(31)=-abb73(31)*abb73(47)
      abb73(38)=-abb73(36)*abb73(21)
      abb73(40)=3.0_ki*abb73(40)+2.0_ki*abb73(43)
      abb73(41)=-2.0_ki*abb73(41)
      abb73(33)=-spbk1e1*abb73(33)
      abb73(14)=-spbk1e1*abb73(14)
      abb73(43)=abb73(36)*abb73(42)
      abb73(44)=-abb73(36)*abb73(25)
      abb73(45)=-abb73(36)*abb73(32)
      abb73(46)=abb73(36)*abb73(39)
      abb73(36)=-abb73(36)*abb73(22)
      R2d73=0.0_ki
      rat2 = rat2 + R2d73
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='73' value='", &
          & R2d73, "'/>"
      end if
   end subroutine
end module p2_gg_httbar_abbrevd73h0_qp
